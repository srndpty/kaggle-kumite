"""2 つの run の小さい差の確認（グループ単位の bootstrap による、同じ行どうしのペア比較）。

同じ系列（店舗など）の行は独立でないので、行ではなくグループを再標本化する。
指標は「行ごとの損失の平均」を変換したもの（RMSE 系なら損失 = 二乗誤差、変換 = sqrt）に限る。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapDiff:
    """差は (other - base)。小さいほど良い指標なら、負が改善。"""

    point: float  # 全行での差
    low: float  # 95% 区間の下端
    high: float  # 95% 区間の上端
    share_negative: float  # 差が負になった再標本の割合


def group_bootstrap_diff(
    base_loss: np.ndarray,
    other_loss: np.ndarray,
    groups: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray] = np.sqrt,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> BootstrapDiff:
    """行ごとの損失（同じ行・同じ並び）から、transform(損失の平均) の差をグループ単位で再標本化する。

    同じ seed なら、どの run を比べても同じ再標本を使う（基準に対する複数の run の差がそろう）。
    """
    if not (base_loss.shape == other_loss.shape == groups.shape) or base_loss.ndim != 1:
        raise ValueError(
            f"損失とグループは同じ長さの 1 次元配列である必要があります（{base_loss.shape}, "
            f"{other_loss.shape}, {groups.shape}）"
        )
    if len(base_loss) == 0:
        raise ValueError("損失が空です")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap は 1 以上である必要があります（{n_bootstrap}）")
    if not (np.isfinite(base_loss).all() and np.isfinite(other_loss).all()):
        raise ValueError("損失に NaN または Inf があります")
    if pd.isna(groups).any():
        raise ValueError("グループ ID に欠損があります")
    codes, uniques = pd.factorize(groups)
    n_groups = len(uniques)
    rng = np.random.default_rng(seed)
    counts = np.bincount(codes, minlength=n_groups)
    base_sums = np.bincount(codes, weights=base_loss, minlength=n_groups)
    other_sums = np.bincount(codes, weights=other_loss, minlength=n_groups)

    diff = np.empty(n_bootstrap)
    # 顧客数が多いときも再標本を一括で確保しない。
    batch_size = max(1, 1_000_000 // n_groups)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        draws = rng.integers(0, n_groups, size=(stop - start, n_groups))
        n = counts[draws].sum(axis=1)
        diff[start:stop] = transform(other_sums[draws].sum(axis=1) / n) - transform(
            base_sums[draws].sum(axis=1) / n
        )
    low, high = np.percentile(diff, [2.5, 97.5])
    total = counts.sum()
    point = transform(np.array(other_sums.sum() / total)) - transform(
        np.array(base_sums.sum() / total)
    )
    return BootstrapDiff(
        point=float(point),
        low=float(low),
        high=float(high),
        share_negative=float((diff < 0).mean()),
    )
