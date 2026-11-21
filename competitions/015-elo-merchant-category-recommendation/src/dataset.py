"""カード単位の特徴量と提出データを組み立てる。"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from kumite.checks import validate_regression

ID = "card_id"
TARGET = "target"
CARD_FILES = ("train.csv", "test.csv", "sample_submission.csv")
TRANSACTION_FILES = ("historical_transactions.csv", "new_merchant_transactions.csv")
TRANSACTION_COLUMNS = [ID, "purchase_amount", "authorized_flag", "merchant_id", "month_lag"]


def load_cards(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (pd.read_csv(data / filename) for filename in CARD_FILES)
    for table in (train, test, sample):
        if table.empty or table[ID].isna().any() or not table[ID].is_unique:
            raise ValueError("カード ID は空でなく一意である必要があります")
    if set(train[ID]) & set(test[ID]):
        raise ValueError("学習とテストでカード ID が重複しています")
    if list(sample.columns) != [ID, TARGET] or set(sample[ID]) != set(test[ID]):
        raise ValueError("公式提出サンプルの列または ID が不正です")
    validate_regression(train[TARGET].to_numpy(), len(train))
    return train, test, sample


def aggregate(transactions: pl.LazyFrame, prefix: str) -> pd.DataFrame:
    amount = pl.col("purchase_amount")
    return pd.DataFrame(
        transactions.group_by(ID)
        .agg(
            pl.len().alias(f"{prefix}_count"),
            amount.sum().alias(f"{prefix}_amount_sum"),
            amount.mean().alias(f"{prefix}_amount_mean"),
            amount.std().alias(f"{prefix}_amount_std"),
            (pl.col("authorized_flag") == "Y").mean().alias(f"{prefix}_approval_rate"),
            pl.col("merchant_id").drop_nulls().n_unique().alias(f"{prefix}_merchants"),
        )
        .collect()
        .to_dict(as_series=False)
    ).set_index(ID)


def features(data: Path, cards: pd.DataFrame, trial: str) -> pd.DataFrame:
    history = pl.scan_csv(data / TRANSACTION_FILES[0]).select(TRANSACTION_COLUMNS)
    aggregates = [aggregate(history, "history")]
    if trial == "E02":
        aggregates.append(aggregate(history.filter(pl.col("month_lag") == 0), "recent"))
    if trial == "E03":
        new = pl.scan_csv(data / TRANSACTION_FILES[1]).select(TRANSACTION_COLUMNS)
        aggregates.append(aggregate(new, "new"))

    if trial == "E05":
        month_counts = history.group_by(ID).agg(
            (pl.col("month_lag") == lag).sum().alias(f"lag_{-lag}_count") for lag in range(-13, 1)
        )
        aggregates.append(
            pd.DataFrame(month_counts.collect().to_dict(as_series=False)).set_index(ID)
        )

    first_month = pd.to_datetime(cards["first_active_month"])
    result = cards[["feature_1", "feature_2", "feature_3"]].copy()
    result["first_active_month"] = first_month.dt.year * 12 + first_month.dt.month
    for table in aggregates:
        aligned = table.reindex(cards[ID]).set_axis(cards.index)
        result = result.join(aligned, validate="one_to_one")
        for column in table.columns:
            if column.endswith(("_count", "_merchants")):
                result[column] = result[column].fillna(0)
    return result


def submission(test: pd.DataFrame, sample: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    validate_regression(prediction, len(test))
    result = sample[[ID]].join(pd.Series(prediction, index=test[ID], name=TARGET), on=ID)
    validate_regression(result[TARGET].to_numpy(), len(sample))
    return result
