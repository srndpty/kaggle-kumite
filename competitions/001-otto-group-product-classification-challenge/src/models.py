"""モデル定義。どのモデルも同じ関数形 `fit_predict` にそろえ、学習ループ (train.py) と分離する。"""

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from dataset import CLASSES
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kumite.crossfit import FoldOutput

# (学習 X, 学習 y, 検証 X, 検証 y, 予測対象 X, params) -> 検証・予測対象への予測確率
FitPredict = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]], FoldOutput
]


def prior(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    """学習 fold のクラス比をそのまま全行の予測にする。"""
    ratio = np.bincount(y_tr, minlength=len(CLASSES)) / len(y_tr)
    return FoldOutput(np.tile(ratio, (len(x_va), 1)), np.tile(ratio, (len(x_te), 1)), {})


def lightgbm(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
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
    return FoldOutput(
        np.asarray(booster.predict(x_va, num_iteration=best)),
        np.asarray(booster.predict(x_te, num_iteration=best)),
        {"best_iteration": best},
    )


def logistic_regression(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    """標準化 + 多クラスロジスティック回帰。標準化は学習 fold だけで fit する。"""
    classifier = LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(x_tr, y_tr)
    return FoldOutput(
        model.predict_proba(x_va),
        model.predict_proba(x_te),
        {"n_iter": int(classifier.n_iter_[0])},
    )


def _mix_uniform(pred: np.ndarray, weight: float) -> np.ndarray:
    """確率 0 を避けるため一様分布を少し混ぜる。kNN・木の投票確率は 0 になりやすく、logloss が発散する。"""
    return (1 - weight) * pred + weight / len(CLASSES)


def knn(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    """標準化 + 距離重み付き kNN。標準化は学習 fold だけで fit する。"""
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights="distance", n_jobs=-1),
    )
    model.fit(x_tr, y_tr)
    w = params["uniform_mix"]
    return FoldOutput(
        _mix_uniform(model.predict_proba(x_va), w), _mix_uniform(model.predict_proba(x_te), w), {}
    )


def extra_trees(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    model = ExtraTreesClassifier(
        n_estimators=params["n_estimators"], n_jobs=-1, random_state=params["seed"]
    )
    model.fit(x_tr, y_tr)
    w = params["uniform_mix"]
    # 多出力分類ではリストを返しうる型定義のため、ndarray に固定する（今回は単一出力）。
    return FoldOutput(
        _mix_uniform(np.asarray(model.predict_proba(x_va)), w),
        _mix_uniform(np.asarray(model.predict_proba(x_te)), w),
        {},
    )


MODELS: dict[str, FitPredict] = {
    "prior": prior,
    "lightgbm": lightgbm,
    "logistic_regression": logistic_regression,
    "knn": knn,
    "extra_trees": extra_trees,
}
