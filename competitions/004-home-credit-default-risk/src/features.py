"""ラベルを使わない顧客内履歴集約。予定日と実績の情報時点を区別する。"""

import numpy as np
import pandas as pd
from dataset import KEY, unique_ids

BUREAU_NUMERIC = [
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "CREDIT_DAY_OVERDUE",
]
PREVIOUS_NUMERIC = ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY"]
INSTALLMENT_KEYS = [KEY, "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
INSTALLMENT_COLUMNS = [
    *INSTALLMENT_KEYS,
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]


def _aggregate(frame: pd.DataFrame, prefix: str, numeric: list[str], time: str) -> pd.DataFrame:
    group = frame.groupby(KEY, sort=True)
    result = group[numeric].agg(["mean", "max", "sum"])
    result.columns = [f"{prefix}_{column}_{stat}" for column, stat in result.columns]
    # 全欠損の金額を0とみなさない。
    for column in numeric:
        result[f"{prefix}_{column}_sum"] = group[column].sum(min_count=1)
    result[f"{prefix}_count"] = group.size()
    for stat in ("min", "max"):
        result[f"{prefix}_{time}_{stat}"] = group[time].agg(stat)
    return result


def bureau_features(frame: pd.DataFrame, *, since: int | None = None) -> pd.DataFrame:
    unique_ids(frame, "SK_ID_BUREAU")
    # 更新日時も観測時点。将来の返済予定日自体はこの条件に使わない。
    frame = frame.loc[(frame["DAYS_CREDIT"] <= 0) & (frame["DAYS_CREDIT_UPDATE"] <= 0)].copy()
    prefix = "bureau"
    if since is not None:
        frame = frame.loc[frame["DAYS_CREDIT"] >= since]
        prefix = "bureau_recent"
    result = _aggregate(frame, prefix, BUREAU_NUMERIC, "DAYS_CREDIT")
    for status in ("Active", "Closed", "Sold", "Bad debt"):
        counts = frame["CREDIT_ACTIVE"].eq(status).groupby(frame[KEY]).sum()
        result[f"{prefix}_status_{status.replace(' ', '_')}_count"] = counts
    return result.reset_index()


def previous_features(frame: pd.DataFrame) -> pd.DataFrame:
    unique_ids(frame, "SK_ID_PREV")
    frame = frame.loc[frame["DAYS_DECISION"] <= 0].copy()
    result = _aggregate(frame, "previous", PREVIOUS_NUMERIC, "DAYS_DECISION")
    for status in ("Approved", "Refused", "Canceled", "Unused offer"):
        counts = frame["NAME_CONTRACT_STATUS"].eq(status).groupby(frame[KEY]).sum()
        result[f"previous_status_{status.replace(' ', '_')}_count"] = counts
    return result.reset_index()


def installment_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[INSTALLMENT_KEYS].isna().any().any():
        raise ValueError("返済キーに欠損があります")
    if frame.groupby("SK_ID_PREV")[KEY].nunique().gt(1).any():
        raise ValueError("過去契約が複数顧客に属しています")
    # 将来の支払実績を使わない。未観測の支払を0で補完して返済済とみなさない。
    past = frame.loc[(frame["DAYS_INSTALMENT"] <= 0) & (frame["DAYS_ENTRY_PAYMENT"] <= 0)].copy()
    group = past.groupby(INSTALLMENT_KEYS, sort=False)
    if group[["AMT_INSTALMENT", "DAYS_INSTALMENT"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("同じ分割返済で予定額または期日が一致しません")
    paid = group.agg(
        due=("DAYS_INSTALMENT", "first"),
        expected=("AMT_INSTALMENT", "first"),
        last_payment=("DAYS_ENTRY_PAYMENT", "max"),
    ).reset_index()
    paid["payment"] = group["AMT_PAYMENT"].sum(min_count=1).to_numpy()
    paid["delay"] = (paid["last_payment"] - paid["due"]).clip(lower=0)
    paid["shortfall"] = (paid["expected"] - paid["payment"]).clip(lower=0)
    for source, target in (("delay", "late"), ("shortfall", "underpaid")):
        paid[target] = paid[source].gt(0).astype(float).where(paid[source].notna())
    group_paid = paid.groupby(KEY, sort=True)
    result = group_paid[["delay", "shortfall"]].agg(["mean", "max"])
    result.columns = [f"installments_{column}_{stat}" for column, stat in result.columns]
    result["installments_count"] = group_paid.size()
    for column in ("late", "underpaid"):
        result[f"installments_{column}_rate"] = group_paid[column].mean()
    return result.reset_index()


def join_history(main: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    unique_ids(main)
    unique_ids(history)
    ids = main[KEY].to_numpy().copy()
    result = main.merge(history, on=KEY, how="left", validate="one_to_one", sort=False)
    if len(result) != len(main) or not np.array_equal(result[KEY], ids):
        raise ValueError("結合で行数または順序が変わりました")
    # 履歴がない場合だけ0。履歴があるが値が欠損する場合は欠損を保つ。
    counts = [str(c) for c in history if str(c).endswith("_count")]
    result[counts] = result[counts].fillna(0)
    return result


def application_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[KEY, "TARGET"], errors="ignore").copy()
    # 公式データの極端な雇用日数は欠損符号として扱う。全実験で共通。
    if "DAYS_EMPLOYED" in result:
        result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(365243, np.nan)
    return result


def encode_categories(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[pd.DataFrame], list[str]]:
    frames = [frame.copy() for frame in (train, valid, test)]
    cats = [str(c) for c in train.select_dtypes(include=["object", "string"]).columns]
    for column in cats:
        dtype = pd.CategoricalDtype(categories=sorted(train[column].dropna().unique()))
        for frame in frames:
            frame[column] = frame[column].where(frame[column].isin(dtype.categories)).astype(dtype)
    for frame in frames:
        numeric = [str(c) for c in frame if c not in cats]
        frame[numeric] = frame[numeric].astype(np.float32)
    return frames, cats
