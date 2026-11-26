"""患者分割と公式 pAUC。"""

import numpy as np
import pandas as pd
from model import SEED
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from kumite.crossfit import Split, split_roles

MAX_FPR = 0.20
SPLIT_VERSION = "patient-v1"


def pauc(y: np.ndarray, prediction: np.ndarray) -> float:
    if y.ndim != 1 or prediction.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("評価には二値ラベルの両クラスと同じ形の予測が必要です")
    if not (np.isfinite(prediction) & (prediction >= 0) & (prediction <= 1)).all():
        raise ValueError("評価用の確率が不正です")
    # 陰陽とスコアの向きを反転し、FPR <= 0.20 の非標準化 AUC を求める。
    fpr, tpr, _ = roc_curve(1 - y, -prediction)
    stop = int(np.searchsorted(fpr, MAX_FPR, side="right"))
    boundary = np.interp(MAX_FPR, fpr[stop - 1 : stop + 1], tpr[stop - 1 : stop + 1])
    return float(auc(np.append(fpr[:stop], MAX_FPR), np.append(tpr[:stop], boundary)))


def make_splits(frame: pd.DataFrame, y: np.ndarray) -> list[Split]:
    patients = frame["patient_id"]
    if patients.isna().any():
        raise ValueError("患者 ID が欠落しています")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(y, y, groups=patients))
    if not ((split_roles(len(y), splits) == 2).sum(axis=0) == 1).all():
        raise ValueError("全病変を一度ずつ検証する必要があります")
    for train, valid in splits:
        if set(patients.iloc[train]) & set(patients.iloc[valid]):
            raise ValueError("患者が学習・検証にまたがっています")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("学習・検証の両方に陽性と陰性が必要です")
    return splits
