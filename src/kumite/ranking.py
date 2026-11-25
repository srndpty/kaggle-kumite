"""順位付き推薦の採点と検査（正解の件数はクエリごとに違ってよい）。"""

import numpy as np


def average_precision(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """クエリごとの AP@K。hits は (クエリ, 順位) の命中、K は列数。正解 0 件のクエリは 0。"""
    if hits.ndim != 2 or n_relevant.shape != (len(hits),):
        raise ValueError(f"命中と正解数の形が一致しません（{hits.shape}, {n_relevant.shape}）")
    k = hits.shape[1]
    found = hits.astype(np.float64)
    precision = found.cumsum(axis=1) / np.arange(1.0, k + 1)
    gains = np.einsum("ij,ij->i", precision, found)
    return gains / np.maximum(1, np.minimum(k, n_relevant))


def sparse_hits(
    ranked: np.ndarray, rows: np.ndarray, items: np.ndarray, n_items: int
) -> np.ndarray:
    """ranked（クエリ×順位の商品番号）が、正解の組 (rows, items) に含まれるか。負の番号は命中しない。"""
    truth = rows.astype(np.int64) * n_items + items
    asked = np.arange(len(ranked), dtype=np.int64)[:, None] * n_items + ranked
    return np.isin(asked, truth) & (ranked >= 0)


def validate_full_rankings(ranked: np.ndarray, shape: tuple[int, int], n_items: int) -> None:
    """全クエリが K 件埋まり、商品番号が範囲内で、クエリ内に重複がないことを確かめる。"""
    if ranked.shape != shape or not np.issubdtype(ranked.dtype, np.integer):
        raise ValueError(f"順位の形または型が不正です（{ranked.shape}, {ranked.dtype}）")
    if ((ranked < 0) | (ranked >= n_items)).any():
        raise ValueError("商品番号が範囲外です")
    ordered = np.sort(ranked, axis=1)
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("重複した商品が含まれます")
