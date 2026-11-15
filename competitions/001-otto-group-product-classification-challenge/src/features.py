"""特徴量セット。いずれも行ごとに完結する変換なので、CV の外で train/test に適用してよい。"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from dataset import FEATURES


def raw(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES]


def log1p(df: pd.DataFrame) -> pd.DataFrame:
    # float32 だと predict_proba も float32 になり、確率の和が 1 からずれて logloss が警告を出す。
    return df[FEATURES].astype("float64").transform(np.log1p)


def row_stats(df: pd.DataFrame) -> pd.DataFrame:
    """生カウントに行合計と非ゼロ数を加える。"""
    x = df[FEATURES]
    return x.assign(row_sum=x.sum(axis=1), row_nnz=(x > 0).sum(axis=1))


FEATURE_SETS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "raw": raw,
    "log1p": log1p,
    "row_stats": row_stats,
}
