"""M5 の公式入力を商品×店舗と日番号でそろえる。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 28
ORIGINS = (1829, 1857, 1885)
PUBLIC_ORIGIN, PRIVATE_ORIGIN = 1913, 1941
KEYS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
COMMON_FILES = ["calendar.csv", "sell_prices.csv"]
VALIDATION_FILE = "sales_train_validation.csv"
EVALUATION_FILE = "sales_train_evaluation.csv"


@dataclass
class Sales:
    series: pd.DataFrame
    units: np.ndarray
    calendar: pd.DataFrame
    prices: np.ndarray
    week_index: np.ndarray

    def price(self, start: int, end: int) -> np.ndarray:
        return self.prices[:, self.week_index[start - 1 : end]]

    def actual(self, start: int, end: int) -> np.ndarray:
        if not 1 <= start <= end <= self.units.shape[1]:
            raise ValueError(f"売上の参照範囲が不正です: {start}–{end}")
        return self.units[:, start - 1 : end]


def load_sales(data_dir: Path, *, evaluation: bool = False) -> Sales:
    end = PRIVATE_ORIGIN if evaluation else PUBLIC_ORIGIN
    sales_file = EVALUATION_FILE if evaluation else VALIDATION_FILE
    days = [f"d_{day}" for day in range(1, end + 1)]
    sales = pd.read_csv(data_dir / sales_file, dtype=dict.fromkeys(days, np.int32))
    series = sales[KEYS].copy()
    if len(series) != 30_490 or series.duplicated(["item_id", "store_id"]).any():
        raise ValueError("商品×店舗の行数またはキーが不正です")
    units = np.ascontiguousarray(sales[days].to_numpy())
    del sales
    calendar = pd.read_csv(data_dir / "calendar.csv")
    if calendar["d"].duplicated().any():
        raise ValueError("暦の日番号が重複しています")
    calendar = calendar.set_index("d").loc[
        [f"d_{day}" for day in range(1, PRIVATE_ORIGIN + HORIZON + 1)]
    ]
    weeks = pd.Index(calendar["wm_yr_wk"].unique())
    price_rows = pd.read_csv(data_dir / "sell_prices.csv", dtype={"sell_price": np.float32})
    price_table = price_rows.pivot(
        index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price"
    )
    series_index = pd.MultiIndex.from_frame(series[["store_id", "item_id"]])
    prices = price_table.reindex(index=series_index, columns=weeks).to_numpy(dtype=np.float32)
    week_index = weeks.get_indexer(pd.Index(calendar["wm_yr_wk"]))
    return Sales(series, units, calendar, prices, week_index)


LEVELS: tuple[tuple[str, ...], ...] = (
    (),
    ("state_id",),
    ("store_id",),
    ("cat_id",),
    ("dept_id",),
    ("state_id", "cat_id"),
    ("state_id", "dept_id"),
    ("store_id", "cat_id"),
    ("store_id", "dept_id"),
    ("item_id",),
    ("item_id", "state_id"),
    ("item_id", "store_id"),
)


@dataclass
class Hierarchy:
    order: np.ndarray
    starts: np.ndarray
    levels: np.ndarray
    labels: pd.DataFrame
    n_bottom: int

    @classmethod
    def from_series(cls, series: pd.DataFrame) -> "Hierarchy":
        groups: list[np.ndarray] = []
        labels: list[pd.DataFrame] = []
        offset = 0
        for level, keys in enumerate(LEVELS, start=1):
            if keys:
                unique, group = np.unique(
                    series[list(keys)].to_numpy(dtype=str), axis=0, return_inverse=True
                )
                names = ["_".join(row) for row in unique]
            else:
                group = np.zeros(len(series), dtype=np.int64)
                names = ["Total_X"]
            groups.append(group + offset)
            labels.append(pd.DataFrame({"level": level, "series": names}))
            offset += len(names)
        index = np.concatenate(groups)
        columns = np.tile(np.arange(len(series)), len(LEVELS))
        order = np.argsort(index, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(index[order])) + 1]
        table = pd.concat(labels, ignore_index=True)
        return cls(columns[order], starts, table["level"].to_numpy() - 1, table, len(series))

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        if values.shape[0] != self.n_bottom or values.ndim not in (1, 2):
            raise ValueError("階層集計の入力形状が不正です")
        # 階層ごとに処理し、全階層分の並べ替え配列を同時に作らない。
        parts: list[np.ndarray] = []
        for start in range(0, len(self.order), self.n_bottom):
            rows = self.order[start : start + self.n_bottom]
            first, last = np.searchsorted(self.starts, [start, start + self.n_bottom])
            parts.append(
                np.add.reduceat(
                    values[rows], self.starts[first:last] - start, axis=0, dtype=np.float64
                )
            )
        return np.concatenate(parts)


def revenue_weights(data: Sales, hierarchy: Hierarchy, origin: int) -> np.ndarray:
    recent = data.actual(origin - HORIZON + 1, origin)
    prices = data.price(origin - HORIZON + 1, origin)
    bad = (recent > 0) & ~np.isfinite(prices)
    if bad.any():
        row, day = np.argwhere(bad)[0]
        raise ValueError(
            f"正の売上に価格がありません: {tuple(data.series.iloc[int(row)])}, d_{origin - HORIZON + 1 + day}"
        )
    revenue = np.sum(recent * np.nan_to_num(prices, nan=0.0), axis=1, dtype=np.float64)
    dollars = hierarchy.aggregate(revenue)
    total = np.bincount(hierarchy.levels, weights=dollars, minlength=len(LEVELS))
    if (total <= 0).any():
        raise ValueError("売上金額が正の系列がない階層があります")
    return dollars / total[hierarchy.levels]


def seasonal(data: Sales, origin: int) -> np.ndarray:
    """直近4週の同曜日平均を4週先まで繰り返す。"""
    history = data.actual(origin - HORIZON + 1, origin)
    return np.tile(history.reshape(len(data.series), 4, 7).mean(axis=1), (1, 4))


def historical_change_scale(
    data: Sales, hierarchy: Hierarchy, origin: int, *, squared: bool
) -> np.ndarray:
    """初回売上以降の日次差分の二乗平均または絶対値平均を計算する。"""
    rows = len(hierarchy.levels)
    first = np.full(rows, origin)
    previous = np.zeros(rows)
    total = np.zeros(rows)
    for start in range(1, origin + 1, 128):
        end = min(origin, start + 127)
        history = hierarchy.aggregate(data.actual(start, end))
        sold = history > 0
        debut = (first == origin) & sold.any(axis=1)
        first[debut] = start - 1 + sold[debut].argmax(axis=1)
        delta = np.diff(history, axis=1, prepend=previous[:, None])
        delta[np.arange(start - 1, end)[None, :] <= first[:, None]] = 0
        total += np.einsum("ij,ij->i", delta, delta) if squared else np.abs(delta).sum(axis=1)
        previous = history[:, -1].copy()
    count = origin - first - 1
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)
