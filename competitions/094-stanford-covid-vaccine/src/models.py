"""位置別の特徴と、位置窓 LightGBM・双方向 GRU。"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from data import BPPS, N_TOKENS, Molecules, bpp, tokens
from torch import nn

from kumite.gbdt import fit_lightgbm_fixed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def position_features(
    molecules: Molecules, use_bpp: bool, bpp_root: Path = BPPS
) -> list[np.ndarray]:
    """分子ごとの (長さ, 特徴数)。

    bpp ありでは、対合確率の行和・最大値と、対合確率で重み付けした相手位置の one-hot を足す。
    """
    result: list[np.ndarray] = []
    for i in range(len(molecules)):
        onehot = tokens(molecules, i)
        if not use_bpp:
            result.append(onehot)
            continue
        matrix = bpp(molecules.ids[i], bpp_root)
        if matrix.shape != (len(onehot), len(onehot)):
            raise ValueError(f"{molecules.ids[i]}: bpp の形が {matrix.shape} です")
        summary = np.stack([matrix.sum(axis=1), matrix.max(axis=1)], axis=1)
        result.append(np.concatenate([onehot, summary, matrix @ onehot], axis=1))
    return result


def n_features(use_bpp: bool) -> int:
    return N_TOKENS + (2 + N_TOKENS if use_bpp else 0)


# --- LightGBM ---------------------------------------------------------------


def window_frame(
    features: Sequence[np.ndarray], positions: int | None, window: int
) -> pd.DataFrame:
    """各位置の前後 `window` 位置の特徴を横に並べる。範囲外は 0。

    `positions` が None なら全位置、整数なら先頭からその数の位置を行にする。
    """
    blocks: list[np.ndarray] = []
    for feature in features:
        length, width = feature.shape
        n = length if positions is None else positions
        padded = np.zeros((length + 2 * window, width))
        padded[window : window + length] = feature
        shifted = [
            padded[window + offset : window + offset + n] for offset in range(-window, window + 1)
        ]
        index = np.arange(n, dtype=np.float64)[:, None]
        blocks.append(np.concatenate([*shifted, index, length - 1 - index], axis=1))
    width = features[0].shape[1]
    names = [f"f{offset}_{k}" for offset in range(-window, window + 1) for k in range(width)]
    return pd.DataFrame(np.concatenate(blocks), columns=[*names, "position", "to_end"])


def fit_predict_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weight: np.ndarray | None,
    queries: Sequence[pd.DataFrame],
    params: dict[str, Any],
) -> list[np.ndarray]:
    """目的ごとに別の模型を学習し、各クエリへ (行, 目的) の予測を返す。"""
    outputs = [np.empty((len(q), y_train.shape[1])) for q in queries]
    for k in range(y_train.shape[1]):
        keep = np.isfinite(y_train[:, k])
        w = None if weight is None else weight[keep, k]
        model = fit_lightgbm_fixed(x_train[keep], y_train[keep, k], params, (), sample_weight=w)
        for out, q in zip(outputs, queries, strict=True):
            out[:, k] = np.asarray(model.predict(q), dtype=np.float64)
    return outputs


# --- GRU ---------------------------------------------------------------------


class GRUNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.gru = nn.GRU(
            hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.head = nn.Linear(2 * hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.proj(x))
        return self.head(out)


@dataclass(frozen=True)
class GRUConfig:
    hidden: int
    layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class TrainGroup:
    """同じ長さの学習分子の組。`x` (分子, 長さ, 特徴)、`y`・`weight` (分子, ラベル位置, 目的)。

    損失はラベルのある先頭 `y.shape[1]` 位置だけで計算する。`weight` は組の中の相対的な重みで、
    バッチごとの正規化で全体の大きさは消える。組全体の寄与は `loss_scale` で変える。
    """

    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    loss_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.weight.shape != self.y.shape or self.y.shape[:1] != self.x.shape[:1]:
            raise ValueError("学習ラベルまたは重みの形が想定外です")
        if not np.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("loss_scale は正の有限値が必要です")
        if self.y.shape[1] > self.x.shape[1]:
            raise ValueError("ラベル位置が配列長を超えています")
        if not np.isfinite(self.y[self.weight > 0]).all():
            raise ValueError("重みが正の学習ラベルに NaN があります")


def batch_loss(
    predicted: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, loss_scale: float
) -> torch.Tensor:
    """バッチ内で正規化した重み付き二乗誤差に、組の係数を掛ける。

    重みを一律に何倍しても正規化で相殺されるので、組の寄与は `loss_scale` だけで変わる。
    """
    weighted = (weight * (predicted - y) ** 2).sum() / weight.sum().clamp_min(1e-12)
    return loss_scale * weighted


def fit_predict_gru(
    groups: Sequence[TrainGroup],
    queries: Sequence[np.ndarray],
    config: GRUConfig,
) -> list[np.ndarray]:
    """重み付き二乗誤差で学習し、クエリ (分子, 長さ, 特徴) ごとに全位置の予測を返す。

    エポック数は固定（検証行を見ない）。長さの違う組はバッチを分け、組が複数のときだけ
    バッチの順序も混ぜる（組が1つなら乱数の消費は従来と同じ）。
    """
    if not groups or len({(g.x.shape[2], g.y.shape[2]) for g in groups}) != 1:
        raise ValueError("学習の組は 1 つ以上、特徴数と目的数が共通である必要があります")
    _seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = GRUNet(
        groups[0].x.shape[2], groups[0].y.shape[2], config.hidden, config.layers, config.dropout
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    per_epoch = sum(int(np.ceil(len(g.x) / config.batch_size)) for g in groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=config.epochs * per_epoch
    )
    tensors = [
        tuple(
            torch.tensor(a, dtype=torch.float32, device=DEVICE)
            for a in (g.x, np.nan_to_num(g.y), g.weight)
        )
        for g in groups
    ]
    model.train()
    for _ in range(config.epochs):
        batches: list[tuple[int, torch.Tensor]] = []
        for k, (x, _, _) in enumerate(tensors):
            order = torch.randperm(len(x), generator=generator).to(DEVICE)
            batches.extend(
                (k, order[s : s + config.batch_size]) for s in range(0, len(x), config.batch_size)
            )
        if len(tensors) > 1:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator)]
        for k, batch in batches:
            x, y, w = tensors[k]
            predicted = model(x[batch])[:, : y.shape[1]]
            loss = batch_loss(predicted, y[batch], w[batch], groups[k].loss_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    with torch.no_grad():
        return [
            model(torch.tensor(q, dtype=torch.float32, device=DEVICE))
            .cpu()
            .numpy()
            .astype(np.float64)
            for q in queries
        ]
