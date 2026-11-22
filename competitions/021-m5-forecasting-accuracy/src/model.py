"""予測起点から28日先までを一括学習する全系列共通モデル。"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from dataset import HORIZON, KEYS, Sales

FIRST_TRAIN_ORIGIN = 1096
STRIDE = 7
SEED = 42
ROUNDS = 300
SERIES_BUCKETS = 16
PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "seed": SEED,
    "num_threads": 8,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
BASE_WIDTH = len(KEYS) + 5
LAG_COUNT = 15
EXTRA_WIDTH = {"base": 0, "calendar": 2, "price": 2, "intermittency": 2, "lags": LAG_COUNT}


def training_origins(origin: int) -> range:
    return range(FIRST_TRAIN_ORIGIN, origin - HORIZON + 1, STRIDE)


def seasonal(data: Sales, origin: int) -> np.ndarray:
    """直近4週の同曜日平均を4週先まで繰り返す。"""
    history = data.actual(origin - HORIZON + 1, origin)
    return np.tile(history.reshape(len(data.series), 4, 7).mean(axis=1), (1, 4))


class Features:
    def __init__(self, data: Sales, group: str) -> None:
        self.data = data
        self.group = group
        self.categories = np.column_stack([pd.Categorical(data.series[key]).codes for key in KEYS])
        self.weekday = data.calendar["wday"].to_numpy()
        self.month = data.calendar["month"].to_numpy()
        self.snap = data.calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy()
        self.state = data.series["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy(dtype=int)
        self.event = data.calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).to_numpy()

    def allocate(self, series_origins: int | None = None) -> np.ndarray:
        count = len(self.data.series) if series_origins is None else series_origins
        return np.empty((count * HORIZON, BASE_WIDTH + EXTRA_WIDTH[self.group]), np.float32)

    def fill(
        self, result: np.ndarray, origin: int, series: slice | np.ndarray = slice(None)
    ) -> None:
        history = self.data.actual(origin - HORIZON + 1, origin)[series]
        days = slice(origin, origin + HORIZON)
        x = result.reshape(len(history), HORIZON, result.shape[1])
        x[:, :, : len(KEYS)] = self.categories[series, None, :]
        x[:, :, 5] = self.weekday[days]
        x[:, :, 6] = self.month[days]
        x[:, :, 7] = np.arange(1, HORIZON + 1)
        x[:, :, 8] = history[:, -7:].mean(axis=1)[:, None]
        x[:, :, 9] = history.mean(axis=1)[:, None]
        if self.group == "calendar":
            x[:, :, 10] = self.event[days]
            x[:, :, 11] = self.snap[days, :][:, self.state[series]].T
        elif self.group == "price":
            recent = self.data.price(origin - HORIZON + 1, origin)[series]
            available = np.isfinite(recent).sum(axis=1)
            mean = np.full(len(recent), np.nan)
            np.divide(np.nansum(recent, axis=1), available, out=mean, where=available > 0)
            planned = self.data.price(origin + 1, origin + HORIZON)[series]
            x[:, :, 10] = planned
            x[:, :, 11] = planned / mean[:, None]
        elif self.group == "intermittency":
            history_to_origin = self.data.actual(1, origin)[series] > 0
            elapsed = history_to_origin[:, ::-1].argmax(axis=1).astype(np.float32)
            elapsed[~history_to_origin.any(axis=1)] = np.nan
            x[:, :, 10] = elapsed[:, None]
            x[:, :, 11] = (history > 0).mean(axis=1)[:, None]
        elif self.group == "lags":
            # 予測の1日目は起点-27、28日目は起点が最も新しいラグになる。
            source = self.data.actual(origin - HORIZON - LAG_COUNT + 2, origin)[series]
            lag_windows = np.lib.stride_tricks.sliding_window_view(source, LAG_COUNT, axis=1)
            x[:, :, BASE_WIDTH:] = lag_windows[:, :, ::-1]


def training_batches(origin: int, series_count: int) -> list[tuple[int, np.ndarray]]:
    """起点ごとに学習系列を持ち回りにし、全系列と28日分のリードタイムを学習に含める。"""
    count = min(SERIES_BUCKETS, series_count)
    buckets = np.array_split(np.random.default_rng(SEED).permutation(series_count), count)
    return [(past, buckets[i % count]) for i, past in enumerate(training_origins(origin))]


def predict(data: Sales, origin: int, group: str) -> np.ndarray:
    features = Features(data, group)
    batches = training_batches(origin, len(data.series))
    blocks = features.allocate(sum(len(series) for _, series in batches))
    labels = np.empty(len(blocks), dtype=np.float32)
    offset = 0
    for past, series in batches:
        end = offset + len(series) * HORIZON
        features.fill(blocks[offset:end], past, series)
        labels[offset:end] = data.actual(past + 1, past + HORIZON)[series].ravel()
        offset = end
    dataset = lgb.Dataset(blocks, labels, categorical_feature=list(range(len(KEYS))))
    booster = lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)
    target = features.allocate()
    features.fill(target, origin)
    prediction = np.asarray(booster.predict(target)).reshape(len(data.series), HORIZON)
    return np.maximum(prediction, 0.0).astype(np.float32)
