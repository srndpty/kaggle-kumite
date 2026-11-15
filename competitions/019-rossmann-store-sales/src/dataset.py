"""Rossmann Store Sales のデータ読み込み。EDA・学習・提出で共通に使う。"""

from pathlib import Path

import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
OUTPUT_DIR = COMPETITION_DIR / "outputs"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
STORE_CSV = DATA_DIR / "store.csv"
DATA_FILES = (TRAIN_CSV, TEST_CSV, STORE_CSV)

TARGET = "Sales"
# train にだけある列。予測時点では分からないので特徴量に使わない。
TRAIN_ONLY = ("Sales", "Customers")
_DTYPES = {"StateHoliday": "str"}  # "0"・"a"・"b"・"c"。数値の 0 と混在して読まれないようにする


def load_train() -> pd.DataFrame:
    """1 行 = 1 店舗 × 1 日。日付・店舗の昇順に並べ替えて返す（元のファイルは日付の降順）。"""
    df = pd.read_csv(TRAIN_CSV, dtype=_DTYPES, parse_dates=["Date"])
    return df.sort_values(["Date", "Store"], kind="stable").reset_index(drop=True)


def modeling_rows(train: pd.DataFrame) -> pd.DataFrame:
    """学習・検証に使う行（売上 > 0）。評価指標は売上 0 の行を無視し、休業日は Open で分かる。"""
    return train[train[TARGET] > 0].reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """提出ファイルと同じ行順（Id 順）のまま返す。"""
    return pd.read_csv(TEST_CSV, dtype=_DTYPES, parse_dates=["Date"])


def load_store() -> pd.DataFrame:
    """1 行 = 1 店舗。店舗の属性（種別・競合店・継続販促）。"""
    return pd.read_csv(STORE_CSV)
