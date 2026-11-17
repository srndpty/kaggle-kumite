"""分割ごとの学習・予測の繰り返し（クロスフィット）。

外側の CV にも、ラベルを使う特徴量（近傍特徴・スタッキングなど）を学習側だけで作る内側の CV にも使う。
分割は 2 通りで渡せる。

- `cross_fit`: 行ごとの fold 番号。全行をちょうど 1 回ずつ検証する（KFold 系）
- `cross_fit_splits`: (学習行, 検証行) の組の並び。時系列分割のように、検証されない行や
  学習に使わない行があってよい

`fit_fold` には行番号だけを渡す。検証行のラベルを読まないことは呼び出し側の責任で、
`kumite.checks.check_label_independence` で確かめる。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# (学習行の番号, 検証行の番号)
Split = tuple[np.ndarray, np.ndarray]
FitFold = Callable[[np.ndarray, np.ndarray], "FoldOutput"]


@dataclass(frozen=True)
class FoldOutput:
    """1 分割の結果。valid は検証行への予測、query は検証以外の予測対象（テストなど）への予測。"""

    valid: np.ndarray
    query: np.ndarray
    info: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class CrossFitResult:
    oof: np.ndarray  # どの分割でも検証されなかった行は NaN
    query: np.ndarray  # 各分割の query 予測の平均
    infos: list[dict[str, Any]]


def folds_to_splits(folds: np.ndarray) -> list[Split]:
    """folds[i] は行 i の検証 fold 番号（0 始まり）。fold k は k 以外の行で学習し、k の行を検証する。"""
    if folds.ndim != 1 or len(folds) == 0:
        raise ValueError(f"folds は空でない 1 次元配列である必要があります（形状 {folds.shape}）")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError(f"folds は整数である必要があります（dtype {folds.dtype}）")
    n_folds = int(folds.max()) + 1
    if n_folds < 2 or set(np.unique(folds).tolist()) != set(range(n_folds)):
        raise ValueError(f"fold 番号は 0 からの連番で 2 つ以上必要です（値 {np.unique(folds)}）")
    return [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in range(n_folds)]


def split_roles(n_rows: int, splits: Sequence[Split]) -> np.ndarray:
    """分割 × 行の役割（0 = 不使用、1 = 学習、2 = 検証）。分割の検査と、同一性のハッシュに使う。

    学習行と検証行が重なる分割、同じ行を 2 回検証する分割は拒否する。
    """
    if n_rows <= 0 or len(splits) == 0:
        raise ValueError(f"行数（{n_rows}）と分割数（{len(splits)}）は 1 以上である必要があります")
    roles = np.zeros((len(splits), n_rows), dtype=np.int8)
    for k, (train_idx, valid_idx) in enumerate(splits):
        for name, idx in (("学習行", train_idx), ("検証行", valid_idx)):
            if idx.ndim != 1 or len(idx) == 0 or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"分割 {k}: {name}は空でない 1 次元の整数配列である必要があります")
            if idx.min() < 0 or idx.max() >= n_rows or len(np.unique(idx)) != len(idx):
                raise ValueError(f"分割 {k}: {name}に範囲外または重複した行番号があります")
        roles[k, train_idx] = 1
        if (roles[k, valid_idx] == 1).any():
            raise ValueError(f"分割 {k}: 学習行と検証行が重なっています")
        roles[k, valid_idx] = 2
    if ((roles == 2).sum(axis=0) > 1).any():
        raise ValueError("同じ行が複数の分割で検証されています")
    return roles


def cross_fit(folds: np.ndarray, fit_fold: FitFold) -> CrossFitResult:
    """fold 番号による分割でクロスフィットする（全行が 1 回ずつ検証される）。"""
    return cross_fit_splits(len(folds), folds_to_splits(folds), fit_fold)


def cross_fit_splits(n_rows: int, splits: Sequence[Split], fit_fold: FitFold) -> CrossFitResult:
    """各分割で fit_fold(学習行, 検証行) を呼び、OOF 予測と query 予測の平均を返す。

    各分割の出力は、valid の行数が検証行数と一致し、valid の列方向の形と query の形が全分割で
    そろい、NaN・Inf を含まないことを検査する（ブロードキャストで形の違いが黙って通らないように）。
    """
    split_roles(n_rows, splits)  # 分割の検査
    oof: np.ndarray | None = None
    query_sum: np.ndarray | None = None
    infos: list[dict[str, Any]] = []
    for k, (train_idx, valid_idx) in enumerate(splits):
        out = fit_fold(train_idx, valid_idx)
        _check_fold_output(k, out, len(valid_idx), oof, query_sum)
        if oof is None or query_sum is None:
            oof = np.full((n_rows, *out.valid.shape[1:]), np.nan, dtype=np.float64)
            query_sum = np.zeros(out.query.shape, dtype=np.float64)
        oof[valid_idx] = out.valid
        query_sum += out.query
        infos.append(out.info)
    assert oof is not None and query_sum is not None
    return CrossFitResult(oof=oof, query=query_sum / len(splits), infos=infos)


def mean_rounds(infos: Sequence[dict[str, Any]]) -> int:
    """early stopping した各分割の best iteration の平均。全行で学習し直すときの round 数に使う。"""
    return round(float(np.mean([info["best_iteration"] for info in infos])))


def _check_fold_output(
    k: int,
    out: FoldOutput,
    n_valid: int,
    oof: np.ndarray | None,
    query_sum: np.ndarray | None,
) -> None:
    if out.valid.ndim == 0 or len(out.valid) != n_valid:
        raise ValueError(
            f"fold {k}: valid の行数 {out.valid.shape} が検証行数 {n_valid} と違います"
        )
    if oof is not None and out.valid.shape[1:] != oof.shape[1:]:
        raise ValueError(f"fold {k}: valid の形 {out.valid.shape} が前の fold と違います")
    if query_sum is not None and out.query.shape != query_sum.shape:
        raise ValueError(f"fold {k}: query の形 {out.query.shape} が前の fold と違います")
    for name, values in (("valid", out.valid), ("query", out.query)):
        if not np.isfinite(values).all():
            raise ValueError(f"fold {k}: {name} に NaN または Inf があります")
