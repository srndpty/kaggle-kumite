"""画像モデル: 凍結した学習済み backbone の特徴 + ロジスティック回帰と、fine-tune。

画像はデコード済みの uint8 配列（`cache.load_array`、(枚数, H, W, 3)）から行番号で読む。
augmentation は GPU 上でバッチごとに行う（画像ごとに別の乱数のアフィン変換と色の変換）。
early stopping はしない（epoch 数は固定。検証 fold で epoch を選ぶ楽観を避ける）。
"""

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from dataset import N_CLASSES, OUTPUT_DIR
from sklearn.linear_model import LogisticRegression
from timm.data.config import resolve_data_config
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIR = OUTPUT_DIR / "features"


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "resnet50.a1_in1k"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    aug: str = "weak"  # "none" / "weak"
    hflip: bool = False  # 左右反転（右手・左手のクラスが入れ替わるので既定は使わない）
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


# --- 入力 ---


def image_batches(images: np.ndarray, rows: np.ndarray, batch_size: int) -> Iterator[torch.Tensor]:
    """rows の順に画像を読み、GPU 上の (B, 3, H, W) の float（0〜1）にする。"""
    for s in range(0, len(rows), batch_size):
        chunk = np.sort(rows[s : s + batch_size])  # memmap は昇順に読む方が速い
        order = np.argsort(np.argsort(rows[s : s + batch_size]))
        x = torch.from_numpy(images[chunk][order])
        if DEVICE.type == "cuda":
            x = x.pin_memory()
        x = x.to(DEVICE, non_blocking=True)
        yield x.permute(0, 3, 1, 2).float().div_(255)


class Normalizer:
    def __init__(self, backbone: str):
        cfg = resolve_data_config({}, model=backbone)
        self.mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def augment(x: torch.Tensor, cfg: ModelConfig, gen: torch.Generator) -> torch.Tensor:
    """画像ごとに別の乱数で変換する（x は 0〜1、(B, 3, H, W)）。

    weak: 拡大切り出し（面積 0.75〜1.0、縦横比 ±8%）、平行移動（切り出しの余白の範囲）、回転 ±10°、
    明るさ・コントラスト・彩度 ×[0.8, 1.2]。hflip: 確率 0.5 で左右反転。
    """
    if cfg.aug not in ("none", "weak"):
        raise ValueError(f"未知の aug: {cfg.aug}")
    b, _, h, w = x.shape

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(b, device=x.device, generator=gen)

    ones, zeros = torch.ones(b, device=x.device), torch.zeros(b, device=x.device)
    sx, sy, tx, ty, angle = ones, ones.clone(), zeros, zeros.clone(), zeros.clone()
    if cfg.aug == "weak":
        area, aspect = uniform(0.75, 1.0), torch.exp(uniform(math.log(0.92), math.log(1.08)))
        sx = torch.clamp(area.sqrt() * aspect.sqrt(), max=1.0)  # 出力の座標 → 入力の座標の倍率
        sy = torch.clamp(area.sqrt() / aspect.sqrt(), max=1.0)
        tx, ty = (1 - sx) * uniform(-1, 1), (1 - sy) * uniform(-1, 1)
        angle = uniform(-10, 10) * math.pi / 180
    if cfg.hflip:
        sx = sx * torch.where(torch.rand(b, device=x.device, generator=gen) < 0.5, -1.0, 1.0)
    if cfg.aug == "none" and not cfg.hflip:
        return x
    cos, sin = torch.cos(angle), torch.sin(angle)
    # 正規化座標（-1〜1）での回転は、縦横比 w/h を補正しないと画像が歪む
    theta = torch.stack(
        [
            torch.stack([cos * sx, -sin * sy * h / w, tx], dim=1),
            torch.stack([sin * sx * w / h, cos * sy, ty], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if cfg.aug == "weak":
        x = x * uniform(0.8, 1.2).view(b, 1, 1, 1)  # 明るさ
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        mean = gray.mean(dim=(2, 3), keepdim=True)
        x = mean + (x - mean) * uniform(0.8, 1.2).view(b, 1, 1, 1)  # コントラスト
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        x = gray + (x - gray) * uniform(0.8, 1.2).view(b, 1, 1, 1)  # 彩度
        x = x.clamp_(0, 1)
    return x


@torch.no_grad()
def _forward_all(
    model: nn.Module, images: np.ndarray, rows: np.ndarray, norm: Normalizer, batch_size: int
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    for x in image_batches(images, rows, batch_size):
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            out = model(norm(x))
        outs.append(out.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64) if outs else np.empty((0, 0))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- fine-tune ---


def finetune(
    cfg: ModelConfig,
    images: np.ndarray,
    train_idx: np.ndarray,
    train_labels: np.ndarray,
    valid_idx: np.ndarray,
    test_images: np.ndarray,
    monitor: Callable[[np.ndarray], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """学習済み backbone を全層 fine-tune する。AdamW + cosine（1 epoch の warmup）、bf16 の autocast。

    検証画像のラベルは受け取らない。monitor（検証の確率 → 損失）は epoch ごとの記録用で、
    学習の内容には影響しない。検証行が 0 行なら（全行で学習し直すとき）検証の確率は空の配列。
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    model = timm.create_model(cfg.backbone, pretrained=True, num_classes=N_CLASSES).to(DEVICE)
    norm = Normalizer(cfg.backbone)
    labels = torch.as_tensor(train_labels, device=DEVICE)
    per_epoch = len(train_idx) // cfg.batch_size  # 端数のバッチは捨てる
    steps, warmup = cfg.epochs * per_epoch, per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            (s + 1) / warmup
            if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        ),
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    valid = np.empty((0, N_CLASSES))
    eval_batch = cfg.batch_size * 4
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(train_idx))[: per_epoch * cfg.batch_size]
        total = torch.zeros((), device=DEVICE)
        for b, x in enumerate(image_batches(images, train_idx[perm], cfg.batch_size)):
            y = labels[
                torch.as_tensor(perm[b * cfg.batch_size : (b + 1) * cfg.batch_size], device=DEVICE)
            ]
            x = norm(augment(x, cfg, gen))
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach()
        record = {"epoch": epoch + 1, "train_loss": float(total) / per_epoch}
        last = epoch == cfg.epochs - 1
        if len(valid_idx) and (monitor is not None or last):
            valid = softmax(_forward_all(model, images, valid_idx, norm, eval_batch))
            if monitor is not None:
                record["valid_logloss"] = monitor(valid)
        history.append(record)
        print(
            f"    epoch {epoch + 1}: {record}（{time.perf_counter() - started:.0f} 秒）", flush=True
        )
    test = softmax(_forward_all(model, test_images, np.arange(len(test_images)), norm, eval_batch))
    return valid, test, {"history": history, "train_sec": time.perf_counter() - started}


# --- 凍結特徴 + ロジスティック回帰 ---


@torch.no_grad()
def backbone_features(backbone: str, images: np.ndarray, name: str) -> np.ndarray:
    """凍結した backbone の pooled 特徴。画像だけから決まる（ラベルを使わない）ので、一度作って保存する。"""
    h, w = images.shape[1:3]
    out = FEATURE_DIR / f"{backbone}_{h}x{w}_{name}.npy"
    if out.exists():
        return np.load(out)
    model = timm.create_model(backbone, pretrained=True, num_classes=0).to(DEVICE)
    feats = _forward_all(model, images, np.arange(len(images)), Normalizer(backbone), 256)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats.astype(np.float32))
    return feats.astype(np.float32)


def linear_probe(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray, c: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """特徴を学習行の平均・標準偏差で標準化し、多クラスのロジスティック回帰（L2、C 固定）。"""
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-6
    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit((x_tr - mean) / std, y_tr)
    valid = clf.predict_proba((x_va - mean) / std) if len(x_va) else np.empty((0, N_CLASSES))
    return valid, clf.predict_proba((x_te - mean) / std), {"n_iter": int(clf.n_iter_.max())}


# --- 指標 ---


def logloss(y: np.ndarray, proba: np.ndarray) -> float:
    """多クラス logloss（Kaggle と同じく確率を [1e-15, 1 - 1e-15] に切り、行の和で割り直す）。"""
    return float(row_logloss(y, proba).mean())


def row_logloss(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-15, 1 - 1e-15)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])
