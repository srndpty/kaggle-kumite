"""モデル定義。どのモデルも同じ関数形 `fit_predict` にそろえ、学習ループ (train.py) と分離する。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from dataset import CLASSES
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FoldResult:
    valid_pred: np.ndarray
    test_pred: np.ndarray
    info: dict[str, Any]


FitPredict = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]], FoldResult
]


def prior(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldResult:
    """学習 fold のクラス比をそのまま全行の予測にする。"""
    ratio = np.bincount(y_tr, minlength=len(CLASSES)) / len(y_tr)
    return FoldResult(np.tile(ratio, (len(x_va), 1)), np.tile(ratio, (len(x_te), 1)), {})


def lightgbm(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldResult:
    """検証 fold で early stopping する（OOF スコアはわずかに楽観側に偏る）。"""
    lgb_params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "verbosity": -1,
        **{k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping")},
    }
    booster = lgb.train(
        lgb_params,
        lgb.Dataset(x_tr, y_tr),
        num_boost_round=params["num_boost_round"],
        valid_sets=[lgb.Dataset(x_va, y_va)],
        callbacks=[lgb.early_stopping(params["early_stopping"], verbose=False)],
    )
    best = booster.best_iteration
    return FoldResult(
        booster.predict(x_va, num_iteration=best),
        booster.predict(x_te, num_iteration=best),
        {"best_iteration": best},
    )


def logistic_regression(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldResult:
    """標準化 + 多クラスロジスティック回帰。標準化は学習 fold だけで fit する。"""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    )
    model.fit(x_tr, y_tr)
    return FoldResult(
        model.predict_proba(x_va),
        model.predict_proba(x_te),
        {"n_iter": int(model[-1].n_iter_[0])},
    )


MODELS: dict[str, FitPredict] = {
    "prior": prior,
    "lightgbm": lightgbm,
    "logistic_regression": logistic_regression,
}
