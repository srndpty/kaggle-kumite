"""似た画像の近傍と、近傍の予測の平均（P6。1 位・3 位の着眼点）。

test 画像は同じ運転手の動画のコマの集まりで、ほぼ同じ画像がある（EDA: test → test の相関の中央値 0.990）。
近傍の予測を平均すれば、1 枚ごとの予測の揺れが均される。近傍は同じ集合（test なら test、CV なら同じ fold の
検証画像）の中だけで探す。画像だけから決まり、ラベルは使わない。
"""

import numpy as np
import torch
from cache import THUMBS_NPZ
from models import DEVICE, backbone_features


def normalized(x: np.ndarray) -> np.ndarray:
    """行ごとに平均 0・ノルム 1 にする（内積が相関になる）。一様な画像は 0 ベクトル。"""
    x = x.reshape(len(x), -1).astype(np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def features(
    kind: str, split: str, images: np.ndarray, backbone: str, source: str
) -> tuple[np.ndarray, dict[str, str]]:
    """近傍を探す特徴と、run に記録する入力（名前 → ハッシュ）。

    thumb: 32x24 のグレースケール小画像（画像キャッシュの manifest に含まれる。source で照合済み）。
    resnet: 凍結 backbone の pooled 特徴（特徴キャッシュの manifest を照合する）。
    """
    if kind == "thumb":
        thumbs = np.load(THUMBS_NPZ)[split]
        if len(thumbs) != len(images):
            raise ValueError(f"小画像の行数が画像と違います（{len(thumbs)} / {len(images)}）")
        return normalized(thumbs), {}
    if kind == "resnet":
        x, digest = backbone_features(backbone, images, split, source)
        return normalized(x), {f"features/{backbone}_{split}": digest}
    raise ValueError(f"未知の特徴: {kind}")


@torch.no_grad()
def knn(x: np.ndarray, k: int, batch: int = 4096) -> np.ndarray:
    """各行に最も近い（内積が最大の）k 行の番号 (行数, k)。自分自身は除く。

    類似度が同じ近傍は、行番号の小さい順に選ぶ（安定ソート）。
    """
    if x.ndim != 2:
        raise ValueError(f"特徴は 2 次元である必要があります（{x.shape}）")
    if not 1 <= k < len(x):
        raise ValueError(f"k は 1 以上、行数 - 1（{len(x) - 1}）以下である必要があります（k={k}）")
    if not np.isfinite(x).all():
        raise ValueError("特徴に NaN・Inf があります")
    ref = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(DEVICE)
    out = np.empty((len(x), k), dtype=np.int64)
    for s in range(0, len(x), batch):
        sim = ref[s : s + batch] @ ref.T
        rows = torch.arange(len(sim), device=DEVICE)
        sim[rows, rows + s] = -torch.inf
        order = torch.sort(sim, dim=1, descending=True, stable=True).indices[:, :k]
        out[s : s + batch] = order.cpu().numpy()
    return out


def smooth(proba: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """自分と k 個の近傍の予測の単純平均。idx は `knn` の出力（同じ行の集合の中の番号、自分を含まない）。"""
    n = len(proba)
    if idx.ndim != 2 or idx.shape[0] != n or idx.shape[1] < 1:
        raise ValueError(
            f"近傍の番号の形状が予測と合いません（予測 {proba.shape}、近傍 {idx.shape}）"
        )
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError("近傍の番号が予測の行の範囲外です")
    if (idx == np.arange(n)[:, None]).any():
        raise ValueError("近傍に自分自身が含まれています")
    return (proba + proba[idx].sum(axis=1)) / (idx.shape[1] + 1)
