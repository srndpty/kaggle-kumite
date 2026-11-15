"""Recruit Restaurant Visitor Forecasting のデータ読み込み。EDA・学習・提出で共通に使う。

店舗 ID は air（Air レジ）と hpg（ホットペッパーグルメ）の 2 系統。予測対象は air の店舗で、
hpg の予約は `store_id_relation.csv` で対応の付く店舗だけ air に結び付けられる。
"""

from pathlib import Path

import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
OUTPUT_DIR = COMPETITION_DIR / "outputs"
VISIT_CSV = DATA_DIR / "air_visit_data.csv"
AIR_RESERVE_CSV = DATA_DIR / "air_reserve.csv"
HPG_RESERVE_CSV = DATA_DIR / "hpg_reserve.csv"
AIR_STORE_CSV = DATA_DIR / "air_store_info.csv"
HPG_STORE_CSV = DATA_DIR / "hpg_store_info.csv"
RELATION_CSV = DATA_DIR / "store_id_relation.csv"
DATE_INFO_CSV = DATA_DIR / "date_info.csv"
SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
DATA_FILES = (
    VISIT_CSV,
    AIR_RESERVE_CSV,
    HPG_RESERVE_CSV,
    AIR_STORE_CSV,
    HPG_STORE_CSV,
    RELATION_CSV,
    DATE_INFO_CSV,
    SUBMISSION_CSV,
)

TARGET = "visitors"
STORE = "air_store_id"
DATE = "visit_date"


def load_visits() -> pd.DataFrame:
    """1 行 = 1 店舗 × 1 日の来店数。来店記録のある日だけ。日付・店舗の昇順で返す。"""
    df = pd.read_csv(VISIT_CSV, parse_dates=[DATE])
    return df.sort_values([DATE, STORE], kind="stable").reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """提出ファイルと同じ行順のまま、id を店舗と日付に分けて返す。"""
    df = pd.read_csv(SUBMISSION_CSV, usecols=["id"])
    parts = df["id"].str.rsplit("_", n=1, expand=True)
    return df.assign(**{STORE: parts[0], DATE: pd.to_datetime(parts[1])})


def load_reserves() -> pd.DataFrame:
    """air と hpg の予約を air の店舗 ID でまとめて返す。対応の付かない hpg の予約は除く。

    1 行 = 1 件の予約。source は "air" / "hpg"。
    """
    times = ["visit_datetime", "reserve_datetime"]
    air = pd.read_csv(AIR_RESERVE_CSV, parse_dates=times).assign(source="air")
    hpg = pd.read_csv(HPG_RESERVE_CSV, parse_dates=times)
    relation = pd.read_csv(RELATION_CSV)
    hpg = hpg.merge(relation, on="hpg_store_id", how="inner").drop(columns="hpg_store_id")
    return pd.concat([air, hpg.assign(source="hpg")], ignore_index=True)


def load_air_stores() -> pd.DataFrame:
    """1 行 = 1 店舗（air）。ジャンル・地域・緯度経度。"""
    return pd.read_csv(AIR_STORE_CSV)


def load_date_info() -> pd.DataFrame:
    """1 行 = 1 日。曜日と祝日フラグ（train と test の期間をすべて含む）。"""
    return pd.read_csv(DATE_INFO_CSV, parse_dates=["calendar_date"])
