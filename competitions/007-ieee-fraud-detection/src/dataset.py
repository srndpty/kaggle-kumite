"""取引の読み込み、時間分割、予測時点で利用できる履歴特徴。"""

from pathlib import Path

import numpy as np
import pandas as pd

from kumite.crossfit import Split
from kumite.timesplit import windows

ID = "TransactionID"
TIME = "TransactionDT"
TARGET = "isFraud"
CARD = ["card1", "card2"]
DAY = 86400
CATEGORICAL = {
    "ProductCD",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"card{i}" for i in range(1, 7)),
    *(f"M{i}" for i in range(1, 10)),
    *(f"id_{i}" for i in range(12, 39)),
}


def check_ids(frame: pd.DataFrame) -> None:
    ids = frame[ID]
    if ids.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError("取引 ID が空、欠損、または重複しています")


def join_identity(trades: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    check_ids(trades)
    check_ids(identity)
    identity = identity.rename(columns=lambda name: name.replace("-", "_"))
    if not identity.columns.is_unique or (set(trades) & set(identity)) != {ID}:
        raise ValueError("本人確認列が重複しています")
    return trades.merge(
        identity.assign(has_identity=True),
        on=ID,
        how="left",
        validate="one_to_one",
    ).assign(has_identity=lambda rows: rows["has_identity"].eq(True))


def load(data: Path, split: str, identity: bool = False) -> pd.DataFrame:
    trades = pd.read_csv(data / f"{split}_transaction.csv")
    check_ids(trades)
    if trades[TIME].isna().any() or not np.isfinite(trades[TIME]).all():
        raise ValueError("取引時刻に欠損または非有限値があります")
    if split == "train" and set(trades[TARGET].unique()) != {0, 1}:
        raise ValueError("学習ラベルは 0 と 1 である必要があります")
    if identity:
        return join_identity(trades, pd.read_csv(data / f"{split}_identity.csv"))
    return trades


def make_splits(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[Split], str]:
    days = pd.to_datetime(train[TIME] // DAY, unit="D")
    gap = max(0, int(test[TIME].min() // DAY - train[TIME].max() // DAY - 1))
    for horizon in (28, 14):
        splits = [
            (
                np.flatnonzero(days < start - pd.Timedelta(days=gap)),
                np.flatnonzero(days.between(start, end)),
            )
            for start, end in windows(days.max(), horizon, 3)
        ]
        if all(train[TARGET].iloc[idx].nunique() == 2 for pair in splits for idx in pair):
            return splits, f"time-v1-{horizon}d-gap{gap}d"
    raise ValueError("14 日窓でも全分割で両クラスを確保できません")


def history_counts(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """参照行だけを数える。窓の下端は含め、同時刻は除外する。"""
    columns = [*CARD, TIME]
    past = reference[columns].dropna().astype(dict.fromkeys(CARD, "float64")).sort_values(TIME)
    past = past.assign(count=past.groupby(CARD, sort=False).cumcount() + 1)
    rows = query[columns].dropna().astype(dict.fromkeys(CARD, "float64")).sort_values(TIME)
    index = rows.index
    counts = []
    for offset in (0, DAY, 7 * DAY):
        joined = pd.merge_asof(
            rows.assign(TransactionDT=rows[TIME] - offset),
            past,
            on=TIME,
            by=CARD,
            allow_exact_matches=False,
        )
        counts.append(pd.Series(joined["count"].fillna(0).to_numpy(), index=index))
    return pd.DataFrame(
        {"card_count_1d": counts[0] - counts[1], "card_count_7d": counts[0] - counts[2]},
    ).reindex(query.index)


def relative_amount(reference: pd.DataFrame, query: pd.DataFrame) -> pd.Series:
    """カードの厳密に過去の平均取引額に対する当該取引額の比率。"""
    amount = "TransactionAmt"
    columns = [*CARD, TIME, amount]
    past = reference[columns].dropna().astype(dict.fromkeys(CARD, "float64")).sort_values(TIME)
    group = past.groupby(CARD, sort=False)
    past["historical_mean"] = group[amount].cumsum() / (group.cumcount() + 1)
    rows = query[columns].dropna().astype(dict.fromkeys(CARD, "float64")).sort_values(TIME)
    joined = pd.merge_asof(
        rows,
        past[[*CARD, TIME, "historical_mean"]],
        on=TIME,
        by=CARD,
        allow_exact_matches=False,
    )
    ratio = joined[amount] / joined["historical_mean"].where(joined["historical_mean"] > 0)
    return pd.Series(ratio.to_numpy(), index=rows.index, name="relative_amount").reindex(
        query.index
    )


def features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    trial: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    matrices = []
    for rows in (train, query):
        x = rows.drop(columns=[ID, TIME, TARGET], errors="ignore")
        if trial == "history":
            x = x.join(history_counts(train, rows))
        if trial == "relative_amount":
            x = x.join(relative_amount(train, rows))
        if trial == "clock":
            radians = rows[TIME] % DAY * (2 * np.pi / DAY)
            x = x.assign(hour_sin=np.sin(radians), hour_cos=np.cos(radians))
        matrices.append(x)
    left, right = matrices
    right = right[left.columns]
    categories = [
        col for col in left if col in CATEGORICAL or not pd.api.types.is_numeric_dtype(left[col])
    ]
    for col in categories:
        values = pd.Categorical(left[col])
        left[col] = values
        right[col] = pd.Categorical(right[col], categories=values.categories)
    return left, right, categories
