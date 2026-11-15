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
    """folds[i] は行 i の検証 fold 番号（0 始まり）。fold k は k 以外の行で学習し、k の行を予測する。"""
    n_folds = int(folds.max()) + 1
    if set(np.unique(folds).tolist()) != set(range(n_folds)):
        raise ValueError(f"fold 番号は 0..{n_folds - 1} の連番である必要があります")
    oof: np.ndarray | None = None
    query_sum: np.ndarray | None = None
    infos: list[dict[str, Any]] = []
    for k in range(n_folds):
        train_idx, valid_idx = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        out = fit_fold(train_idx, valid_idx)
        if oof is None or query_sum is None:
            oof = np.zeros((len(folds), *out.valid.shape[1:]), dtype=np.float64)
            query_sum = np.zeros(out.query.shape, dtype=np.float64)
        oof[valid_idx] = out.valid
        query_sum += out.query
        infos.append(out.info)
    assert oof is not None and query_sum is not None
    return CrossFitResult(oof=oof, query=query_sum / n_folds, infos=infos)
