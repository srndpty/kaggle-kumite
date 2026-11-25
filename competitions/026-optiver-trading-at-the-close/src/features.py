"""取引日・秒位置で観測可能な入力だけを特徴量に変換する。"""

import lightgbm as lgb
import numpy as np
import pandas as pd

KEYS = ["date_id", "seconds_in_bucket", "stock_id"]
BASE = [
    "stock_id",
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
]
SIGNALS = ["wap", "imbalance_size"]
PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbosity": -1,
}
ROUNDS = 1000


def feature_matrix(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    x = rows[BASE].copy()
    if method == "E01":
        x["spread"] = rows["ask_price"] - rows["bid_price"]
        total_size = rows["bid_size"] + rows["ask_size"]
        x["book_ratio"] = (rows["bid_size"] - rows["ask_size"]) / total_size.where(total_size > 0)
        auction_size = rows["imbalance_size"] + rows["matched_size"]
        x["auction_ratio"] = rows["imbalance_size"] / auction_size.where(auction_size > 0)
    elif method == "E02":
        indexed = rows.set_index(KEYS)[SIGNALS]
        for lag in (10, 60):
            keys = rows[KEYS].copy()
            keys["seconds_in_bucket"] -= lag
            previous = indexed.reindex(pd.MultiIndex.from_frame(keys))
            for name in SIGNALS:
                x[f"{name}_delta_{lag}"] = rows[name].to_numpy() - previous[name].to_numpy()
    elif method == "E03":
        grouped = rows.groupby(KEYS[:2], sort=False)[SIGNALS]
        median = grouped.transform("median")
        rank = grouped.rank(pct=True)
        for name in SIGNALS:
            x[f"{name}_median_delta"] = rows[name] - median[name]
            x[f"{name}_rank"] = rank[name]
    elif method == "E04":
        for second in (240, 300, 420, 480):
            x[f"after_{second}"] = rows["seconds_in_bucket"] >= second
    elif method != "E00":
        raise ValueError(f"Unknown experiment: {method}")
    return x.astype(np.float32)


def fit_model(x: pd.DataFrame, y: np.ndarray) -> lgb.Booster:
    dataset = lgb.Dataset(x, label=y, categorical_feature=["stock_id"])
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


class Replay:
    """現在の取引日の直近60秒だけを保持する。"""

    def __init__(self, method: str) -> None:
        self.method = method
        self.history: pd.DataFrame | None = None
        self.last_batch: tuple[int, int] | None = None

    def transform(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch = batch[list(dict.fromkeys([*KEYS, *BASE]))].reset_index(drop=True)
        ((day, second),) = batch[KEYS[:2]].drop_duplicates().itertuples(index=False, name=None)
        if self.last_batch is not None and (day, second) <= self.last_batch:
            raise ValueError("Batches must be strictly chronological")
        past = self.history if self.last_batch is not None and day == self.last_batch[0] else None
        if past is not None:
            past = past.loc[past["seconds_in_bucket"] >= second - 60]
        visible = batch if past is None else pd.concat([past, batch], ignore_index=True)
        features = feature_matrix(visible, self.method).tail(len(batch)).reset_index(drop=True)
        self.history = visible if self.method == "E02" else None
        self.last_batch = (day, second)
        return features


def prediction_frame(batch: pd.DataFrame, sample: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    ids = batch["row_id"].astype("string").reset_index(drop=True)
    if (
        list(sample.columns) != ["row_id", "target"]
        or ids.isna().any()
        or not ids.is_unique
        or not ids.equals(sample["row_id"].astype("string").reset_index(drop=True))
        or values.shape != (len(ids),)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid predictions or submission IDs")
    result = sample.copy()
    result["target"] = values
    return result
