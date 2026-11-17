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


def features(kind: str, split: str, images: np.ndarray, backbone: str) -> np.ndarray:
    """近傍を探す特徴。thumb: 32x24 のグレースケール小画像、resnet: 凍結 backbone の pooled 特徴。"""
    if kind == "thumb":
        return normalized(np.load(THUMBS_NPZ)[split])
    if kind == "resnet":
        return normalized(backbone_features(backbone, images, split))
    raise ValueError(f"未知の特徴: {kind}")


@torch.no_grad()
def knn(x: np.ndarray, k: int, batch: int = 4096) -> np.ndarray:
    """各行に最も近い（内積が最大の）k 行の番号 (行数, k)。自分自身は除く。"""
    ref = torch.from_numpy(x).to(DEVICE)
    out = np.empty((len(x), k), dtype=np.int64)
    for s in range(0, len(x), batch):
        sim = ref[s : s + batch] @ ref.T
        rows = torch.arange(len(sim), device=DEVICE)
        sim[rows, rows + s] = -torch.inf
        out[s : s + batch] = sim.topk(k, dim=1).indices.cpu().numpy()
    return out


def smooth(proba: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """自分と k 個の近傍の予測の単純平均。"""
    return (proba + proba[idx].sum(axis=1)) / (idx.shape[1] + 1)
