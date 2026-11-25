"""特徴量、時間分割、週別 Gini 安定性指標。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

ID = "case_id"
DATE = "date_decision"
WEEK = "WEEK_NUM"
CREATED = "creationdate_885D"
AMOUNT = "credamount_590A"
HISTORY = ("previous_count", "previous_age_days")
MAX_AMOUNT = "previous_max_amount"
TABLES = ("base", "static_0", "static_cb_0", "applprev_1")


class Stability(TypedDict):
    score: float
    mean_gini: float
    falling_penalty: float
    residual_std: float
    weekly: dict[str, float]


def shards(data: Path, part: str, table: str) -> list[Path]:
    directory = data / "parquet_files" / part
    stem = f"{part}_{table}"
    paths = sorted(directory.glob(f"{stem}_*.parquet"))
    single = directory / f"{stem}.parquet"
    if single.is_file():
        paths.append(single)
    expected = {
        ("train", "static_0"): 2,
        ("test", "static_0"): 3,
        ("train", "applprev_1"): 2,
        ("test", "applprev_1"): 3,
    }
    if len(paths) != expected.get((part, table), 1):
        raise ValueError(f"{directory}: {stem} のファイル数が公式の構成と一致しません")
    return paths


def history_features(
    base: pl.LazyFrame, history: pl.LazyFrame, *, max_amount: bool = False
) -> pl.LazyFrame:
    aggregates = [
        pl.len().alias(HISTORY[0]),
        (pl.col(DATE) - pl.col(CREATED)).min().dt.total_days().alias(HISTORY[1]),
    ]
    if max_amount:
        aggregates.append(pl.col(AMOUNT).max().alias(MAX_AMOUNT))
    return (
        history.join(base.select(ID, DATE), on=ID, how="inner")
        .filter(pl.col(CREATED) < pl.col(DATE))
        .group_by(ID)
        .agg(aggregates)
    )


def load(data: Path, part: str, *, history: bool, max_amount: bool = False) -> pl.DataFrame:
    base = pl.scan_parquet(shards(data, part, "base")).with_columns(
        pl.col(DATE).cast(pl.String).str.to_date("%Y-%m-%d")
    )
    meta = [ID, DATE, WEEK, "target"] if part == "train" else [ID, DATE]
    frame = base.select(meta)
    for table in ("static_0", "static_cb_0"):
        static = pl.scan_parquet(shards(data, part, table))
        columns = []
        for name, dtype in static.collect_schema().items():
            if name in (*meta, "MONTH", WEEK) or name.endswith("D"):
                continue
            column = pl.col(name)
            if dtype != pl.String:
                column = column.cast(pl.Float32)
            columns.append(column.alias(f"{table}__{name}"))
        frame = frame.join(
            static.select(ID, *columns), on=ID, how="left", validate="1:1", maintain_order="left"
        )
    if history:
        past = pl.scan_parquet(shards(data, part, "applprev_1")).select(
            ID,
            pl.col(CREATED).cast(pl.String).str.to_date("%Y-%m-%d"),
            *([pl.col(AMOUNT).cast(pl.Float32)] if max_amount else []),
        )
        frame = frame.join(
            history_features(base, past, max_amount=max_amount),
            on=ID,
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(pl.col(HISTORY[0]).fill_null(0))
    return frame.collect()


def fold_weeks(weeks: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    end = int(weeks.max()) + 1
    splits = []
    for start in range(end - 39, end, 13):
        train = np.flatnonzero(weeks < start)
        valid = np.flatnonzero((weeks >= start) & (weeks < start + 13))
        if set(np.unique(target[train])) != {0, 1}:
            raise ValueError(f"週 {start} より前の学習ラベルに両クラスが必要です")
        for week in range(start, start + 13):
            if set(np.unique(target[weeks == week])) != {0, 1}:
                raise ValueError(f"検証週 {week} のラベルに両クラスが必要です")
        splits.append((train, valid))
    return splits


def stability(weeks: np.ndarray, target: np.ndarray, pred: np.ndarray) -> Stability:
    unique = np.unique(weeks)
    if len(unique) < 2:
        raise ValueError("安定性の計算には2週以上必要です")
    order = np.argsort(weeks, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(weeks[order])) + 1)
    if any(set(np.unique(target[rows])) != {0, 1} for rows in groups):
        raise ValueError("週別 Gini の計算には各週に両クラスが必要です")
    ginis = np.array([2 * roc_auc_score(target[rows], pred[rows]) - 1 for rows in groups])
    x = np.arange(len(unique))
    slope, intercept = np.polyfit(x, ginis, 1)
    mean = float(ginis.mean())
    penalty = 88 * min(0.0, float(slope))
    residual = float(np.std(ginis - (slope * x + intercept)))
    return {
        "score": mean + penalty - 0.5 * residual,
        "mean_gini": mean,
        "falling_penalty": penalty,
        "residual_std": residual,
        "weekly": dict(zip(unique.astype(str).tolist(), ginis.tolist(), strict=True)),
    }


def model_inputs(
    train: pl.DataFrame, query: pl.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train: dict[str, object] = {}
    x_query: dict[str, object] = {}
    categorical = []
    for name in columns:
        a, b = train[name], query[name]
        if a.dtype == pl.String:
            values = pd.Categorical(a.to_numpy())
            x_train[name] = values
            x_query[name] = encode_category(b, values.categories.tolist())
            categorical.append(name)
        else:
            x_train[name] = a.cast(pl.Float32).to_numpy()
            x_query[name] = b.cast(pl.Float32).to_numpy()
    return pd.DataFrame(x_train, copy=False), pd.DataFrame(x_query, copy=False), categorical


def encode_category(column: pl.Series, categories: list[str]) -> pd.Categorical[str]:
    index = pd.Index(categories)
    codes = index.get_indexer(pd.Index(column.cast(pl.String).to_numpy()))
    return pd.Categorical.from_codes(codes, categories=index)


PARAMS: dict[str, Any] = {
    "objective": "binary",
    "seed": 42,
    "num_threads": 8,
    "deterministic": True,
    "force_col_wise": True,
    "num_boost_round": 500,
    "early_stopping": None,
}


def fit_predict(train: pl.DataFrame, query: pl.DataFrame, trial: str) -> np.ndarray:
    columns = [name for name in train.columns if name not in (ID, DATE, WEEK, "target")]
    weeks = train[WEEK].to_numpy()
    if trial == "E03":
        cut = (int(weeks.min()) + int(weeks.max()) + 1) // 2
        before = train.filter(pl.col(WEEK) < cut).select(columns).null_count().row(0)
        after = train.filter(pl.col(WEEK) >= cut).select(columns).null_count().row(0)
        n_before = int(np.sum(weeks < cut))
        columns = [
            name
            for name, early, late in zip(columns, before, after, strict=True)
            if abs(early / n_before - late / (len(weeks) - n_before)) < 0.20
        ]
    x_train, x_query, categorical = model_inputs(train, query, columns)
    weights = None
    if trial == "E02":
        weights = np.exp2((weeks - weeks.max()) / 26)
        weights /= weights.mean()
    dataset = lgb.Dataset(
        x_train, train["target"].to_numpy(), weight=weights, categorical_feature=categorical
    )
    settings = {k: v for k, v in PARAMS.items() if k not in ("num_boost_round", "early_stopping")}
    model = lgb.train(
        {"verbosity": -1, **settings}, dataset, num_boost_round=PARAMS["num_boost_round"]
    )
    return np.asarray(model.predict(x_query), dtype=np.float64)
