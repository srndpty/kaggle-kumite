"""Otto Group のデータ読み込み。EDA・学習・提出で共通に使う。"""

from pathlib import Path

import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
OUTPUT_DIR = COMPETITION_DIR / "outputs"

N_FEATURES = 93
FEATURES = [f"feat_{i}" for i in range(1, N_FEATURES + 1)]
CLASSES = [f"Class_{i}" for i in range(1, 10)]
TARGET = "target"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv", dtype={c: "int16" for c in FEATURES})


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv", dtype={c: "int16" for c in FEATURES})


def encode_target(target: pd.Series) -> pd.Series:
    """`Class_k` を 0 始まりの整数ラベルにする。"""
    return target.map({c: i for i, c in enumerate(CLASSES)}).astype("int8")
