"""モデル定義。どのモデルも同じ関数形にそろえ、学習ループ (pipeline.py) と分離する。

目的変数の変換（log1p など）は pipeline.py が行う。モデルは渡された尺度のまま学習・予測する。
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from features import CATEGORICAL

from kumite.crossfit import FoldOutput
from kumite.gbdt import lightgbm_regression

# (学習 X, 学習 y, 検証 X, 検証 y, 予測対象 X, params) -> 検証・予測対象への予測
FitPredict = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]], FoldOutput
]

_MEDIAN_KEYS = (["store", "dow"], ["store"])


def group_median(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    """店舗 × 曜日ごとの学習期間の中央値。学習期間に無い組は店舗の中央値、それも無ければ全体。"""
    ref = x_tr.assign(_y=y_tr)

    def predict(x: pd.DataFrame) -> np.ndarray:
        pred = np.full(len(x), np.nan)
        for keys in _MEDIAN_KEYS:
            medians = ref.groupby(keys)["_y"].median().rename("_m").reset_index()
            looked_up = x[keys].merge(medians, on=keys, how="left")["_m"].to_numpy()
            pred = np.where(np.isnan(pred), looked_up, pred)
        return np.where(np.isnan(pred), float(np.median(y_tr)), pred)

    return FoldOutput(predict(x_va), predict(x_te), {})


def lightgbm(x_tr, y_tr, x_va, y_va, x_te, params) -> FoldOutput:
    return lightgbm_regression(x_tr, y_tr, x_va, y_va, x_te, params, CATEGORICAL)


MODELS: dict[str, FitPredict] = {"group_median": group_median, "lightgbm": lightgbm}
