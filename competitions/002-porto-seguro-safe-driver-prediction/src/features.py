"""ラベルを使わず、学習 fold 内の辞書でカテゴリを変換する。"""

import numpy as np
import pandas as pd


def build_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    missing_nan: bool = False,
    missing_count: bool = False,
    categorical: bool = True,
    drop_calc: bool = False,
) -> tuple[list[pd.DataFrame], list[str]]:
    frames = [r.copy() for r in (train, valid, test)]
    cats = [str(c) for c in train.columns if str(c).endswith("_cat")]
    for frame in frames:
        if missing_count:
            frame["missing_count"] = ((frame == -1) | frame.isna()).sum(axis=1)
        if missing_nan:
            numeric = [str(c) for c in train.columns if c not in cats]
            frame[numeric] = frame[numeric].replace(-1, np.nan)
    if categorical:
        for c in cats:
            # -1 もカテゴリとして保持する。未知値は欠損扱い。
            dtype = pd.CategoricalDtype(categories=sorted(train[c].dropna().unique()))
            for frame in frames:
                frame[c] = frame[c].where(frame[c].isin(dtype.categories)).astype(dtype)
    if drop_calc:
        frames = [f.drop(columns=[c for c in f if str(c).startswith("ps_calc_")]) for f in frames]
        cats = [c for c in cats if not str(c).startswith("ps_calc_")]
    return frames, cats if categorical else []
