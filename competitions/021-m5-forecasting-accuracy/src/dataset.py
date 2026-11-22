"""M5 の公式入力を商品×店舗と日番号でそろえる。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kumite.checks import validate_regression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
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


def load_sales(*, evaluation: bool = False, data_dir: Path = DATA) -> Sales:
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


def validate_prediction(prediction: np.ndarray, rows: int) -> None:
    if prediction.shape != (rows, HORIZON):
        raise ValueError(f"予測の形状が不正です: {prediction.shape}")
    validate_regression(prediction.ravel(), rows * HORIZON, lower=0)


def submission(
    public: Sales,
    public_pred: np.ndarray,
    private: Sales,
    private_pred: np.ndarray,
    data_dir: Path = DATA,
) -> pd.DataFrame:
    columns = [f"F{day}" for day in range(1, HORIZON + 1)]
    parts = []
    for data, prediction, suffix in (
        (public, public_pred, "validation"),
        (private, private_pred, "evaluation"),
    ):
        validate_prediction(prediction, len(data.series))
        ids = data.series["item_id"] + "_" + data.series["store_id"] + "_" + suffix
        parts.append(pd.DataFrame(prediction, index=pd.Index(ids, name="id"), columns=columns))
    predictions = pd.concat(parts, verify_integrity=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv", usecols=["id"])
    if (
        len(sample) != 2 * len(public.series)
        or set(sample["id"]) != set(predictions.index)
        or sample["id"].duplicated().any()
    ):
        raise ValueError("サンプルと提出 ID が一致しません")
    return predictions.loc[sample["id"]].reset_index()
