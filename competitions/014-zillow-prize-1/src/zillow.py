"""年別の物件結合と時系列検証に共通の前処理。"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from kumite.crossfit import Split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLE = DATA / "sample_submission.csv"
YEARS = (2016, 2017)
TRANSACTIONS = {2016: DATA / "train_2016_v2.csv", 2017: DATA / "train_2017.csv"}
PROPERTIES = {year: DATA / f"properties_{year}.csv" for year in YEARS}
NUMERIC = (
    "bathroomcnt",
    "bedroomcnt",
    "calculatedfinishedsquarefeet",
    "garagecarcnt",
    "latitude",
    "longitude",
    "lotsizesquarefeet",
    "roomcnt",
    "unitcnt",
    "yearbuilt",
    "structuretaxvaluedollarcnt",
    "taxvaluedollarcnt",
    "landtaxvaluedollarcnt",
    "taxamount",
)
TAX_FEATURES = (
    "structuretaxvaluedollarcnt",
    "taxvaluedollarcnt",
    "landtaxvaluedollarcnt",
    "taxamount",
)
CATEGORICAL = ("fips", "propertylandusetypeid", "regionidcity", "regionidzip")
FEATURES = (*NUMERIC, *CATEGORICAL, "month")
ID = "parcelid"
DATE = "transactiondate"
TARGET = "logerror"


def read_properties(year: int) -> pd.DataFrame:
    frame = pd.read_csv(
        PROPERTIES[year],
        usecols=[ID, *NUMERIC, *CATEGORICAL],
        dtype={name: "float32" for name in (*NUMERIC, *CATEGORICAL)},
    )
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("物件スナップショットの ID が欠損または重複しています")
    return frame.set_index(ID)


def join_properties(rows: pd.DataFrame, properties: pd.DataFrame, year: int) -> pd.DataFrame:
    if not rows[DATE].dt.year.eq(year).all():
        raise ValueError("取引年と物件スナップショットの年が一致しません")
    return rows.join(properties, on=ID, validate="many_to_one")


def load_train() -> pd.DataFrame:
    frames = []
    for year, path in TRANSACTIONS.items():
        rows = pd.read_csv(path, parse_dates=[DATE])
        if rows[ID].isna().any() or not np.isfinite(rows[TARGET]).all():
            raise ValueError("取引 ID またはラベルが不正です")
        properties = read_properties(year)
        rows["property_found"] = rows[ID].isin(properties.index)
        frames.append(join_properties(rows, properties, year))
        del properties
    return pd.concat(frames, ignore_index=True)


def check_transaction_order(train: pd.DataFrame) -> None:
    columns = [ID, DATE, TARGET]
    expected = pd.concat(
        [pd.read_csv(path, usecols=columns, parse_dates=[DATE]) for path in TRANSACTIONS.values()],
        ignore_index=True,
    )
    if not train[columns].reset_index(drop=True).equals(expected[columns]):
        raise ValueError("取引行の順序または値が公式 CSV と一致しません")


def time_splits(dates: pd.Series) -> list[Split]:
    boundaries = pd.date_range("2017-01-01", "2017-09-01", freq="2MS")
    return [
        (np.flatnonzero(dates.lt(start)), np.flatnonzero(dates.ge(start) & dates.lt(stop)))
        for start, stop in pairwise(boundaries)
    ]


def feature_frame(
    rows: pd.DataFrame,
    categories: dict[str, pd.CategoricalDtype],
    month: int | None = None,
) -> pd.DataFrame:
    frame = rows.loc[:, [*NUMERIC, *CATEGORICAL]].astype(categories)
    frame["month"] = rows[DATE].dt.month if month is None else month
    return frame.loc[:, list(FEATURES)]


def category_types(rows: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    return {name: pd.CategoricalDtype(sorted(rows[name].dropna().unique())) for name in CATEGORICAL}


def describe(train: pd.DataFrame) -> None:
    print(train[TARGET].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    print(
        "物件結合率・再取引行率", train["property_found"].mean(), train[ID].duplicated(False).mean()
    )
    month = train[DATE].dt.to_period("M")
    print(train.groupby([month, "fips"], dropna=False)[TARGET].agg(["size", "median"]).to_string())
