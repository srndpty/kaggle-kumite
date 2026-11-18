"""LightGBM の回帰・二値分類（検証行での early stopping と、round 数を固定した学習し直し）。

params の `num_boost_round` と `early_stopping` 以外は LightGBM にそのまま渡す。
`early_stopping` が None なら、検証行を見ずに `num_boost_round` まで学習する（全行で学習し直すとき）。
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from kumite.crossfit import FoldOutput

_CONTROL_KEYS = ("num_boost_round", "early_stopping")


class _Train(Protocol):
    """`lgb.train` のうち使う引数だけの型。lightgbm 自身の注釈は callbacks の型が不完全
    （`List[Callable]`）で、strict の型検査を通らないため、ここで型を与える。"""

    def __call__(
        self,
        params: dict[str, Any],
        train_set: lgb.Dataset,
        num_boost_round: int = ...,
        valid_sets: list[lgb.Dataset] | None = ...,
        callbacks: list[Callable[..., None]] | None = ...,
    ) -> lgb.Booster: ...


# 属性を getattr で Any として受け、_Train の型を付ける（直接参照すると lightgbm の注釈の型になる）
_train: _Train = getattr(lgb, "train")  # noqa: B009


def _predict(booster: lgb.Booster, x: pd.DataFrame, num_iteration: int | None) -> np.ndarray:
    if len(x) == 0:  # 全行で学習し直すときは検証行が無い
        return np.zeros(0)
    return np.asarray(booster.predict(x, num_iteration=num_iteration), dtype=np.float64)


def _lightgbm(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    params: dict[str, Any],
    categorical: Sequence[str] = (),
) -> FoldOutput:
    """early stopping するときの CV スコアは、検証行で round 数を選ぶぶん楽観側に偏る。

    categorical のうち x_tr にある列を LightGBM のカテゴリ変数として渡す。
    """
    lgb_params: dict[str, Any] = {
        "verbosity": -1,
        **{k: v for k, v in params.items() if k not in _CONTROL_KEYS},
    }
    used = [c for c in categorical if c in x_tr.columns]
    train_set = lgb.Dataset(x_tr, y_tr, categorical_feature=used)
    if params["early_stopping"] is None:
        rounds = int(params["num_boost_round"])
        booster = _train(lgb_params, train_set, num_boost_round=rounds)
        return FoldOutput(
            _predict(booster, x_va, None), _predict(booster, x_te, None), {"rounds": rounds}
        )
    booster = _train(
        lgb_params,
        train_set,
        num_boost_round=int(params["num_boost_round"]),
        valid_sets=[lgb.Dataset(x_va, y_va, reference=train_set)],
        callbacks=[lgb.early_stopping(int(params["early_stopping"]), verbose=False)],
    )
    best = int(booster.best_iteration)
    return FoldOutput(
        _predict(booster, x_va, best), _predict(booster, x_te, best), {"best_iteration": best}
    )


def lightgbm_regression(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    params: dict[str, Any],
    categorical: Sequence[str] = (),
) -> FoldOutput:
    """回帰。early stopping と固定 round の両方を扱う。"""
    return _lightgbm(
        x_tr, y_tr, x_va, y_va, x_te, {"objective": "regression", **params}, categorical
    )


def lightgbm_binary(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    params: dict[str, Any],
    categorical: Sequence[str] = (),
) -> FoldOutput:
    """二値分類の陽性確率。AUC で early stopping する。"""
    for name, y in (("学習", y_tr), ("検証", y_va)):
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{name}ラベルは 0/1 に限ります")
    if len(np.unique(y_tr)) != 2:
        raise ValueError("学習ラベルには両クラスが必要です")
    return _lightgbm(
        x_tr,
        y_tr,
        x_va,
        y_va,
        x_te,
        {**params, "objective": "binary", "metric": "auc"},
        categorical,
    )
