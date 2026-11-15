"""fold 単位の学習・予測の繰り返し（クロスフィット）。

外側の CV にも、ラベルを使う特徴量（近傍特徴・スタッキングなど）を学習側だけで作る内側の CV にも使う。
`fit_fold` には行番号だけを渡す。検証行のラベルを読まないことは呼び出し側の責任で、
`kumite.checks.check_label_independence` で確かめる。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FoldOutput:
    """1 fold の結果。valid は検証行への予測、query は検証以外の予測対象（テストなど）への予測。"""

    valid: np.ndarray
    query: np.ndarray
    info: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class CrossFitResult:
    oof: np.ndarray
    query: np.ndarray  # 各 fold の query 予測の平均
    infos: list[dict[str, Any]]


def cross_fit(
    folds: np.ndarray, fit_fold: Callable[[np.ndarray, np.ndarray], FoldOutput]
) -> CrossFitResult:
    """folds[i] は行 i の検証 fold 番号（0 始まり）。fold k は k 以外の行で学習し、k の行を予測する。

    各 fold の出力は、valid の行数が検証行数と一致し、valid の列方向の形と query の形が全 fold で
    そろい、NaN・Inf を含まないことを検査する（ブロードキャストで形の違いが黙って通らないように）。
    """
    if folds.ndim != 1 or len(folds) == 0:
        raise ValueError(f"folds は空でない 1 次元配列である必要があります（形状 {folds.shape}）")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError(f"folds は整数である必要があります（dtype {folds.dtype}）")
    n_folds = int(folds.max()) + 1
    if n_folds < 2 or set(np.unique(folds).tolist()) != set(range(n_folds)):
        raise ValueError(f"fold 番号は 0 からの連番で 2 つ以上必要です（値 {np.unique(folds)}）")
    oof: np.ndarray | None = None
    query_sum: np.ndarray | None = None
    infos: list[dict[str, Any]] = []
    for k in range(n_folds):
        train_idx, valid_idx = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        out = fit_fold(train_idx, valid_idx)
        _check_fold_output(k, out, len(valid_idx), oof, query_sum)
        if oof is None or query_sum is None:
            oof = np.zeros((len(folds), *out.valid.shape[1:]), dtype=np.float64)
            query_sum = np.zeros(out.query.shape, dtype=np.float64)
        oof[valid_idx] = out.valid
        query_sum += out.query
        infos.append(out.info)
    assert oof is not None and query_sum is not None
    return CrossFitResult(oof=oof, query=query_sum / n_folds, infos=infos)


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
