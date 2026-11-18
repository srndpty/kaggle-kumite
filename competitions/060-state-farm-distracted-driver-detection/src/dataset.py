"""State Farm Distracted Driver Detection のデータ読み込み。EDA・学習・提出で共通に使う。

学習画像は運転手（subject）ごとの動画のコマで、`driver_imgs_list.csv` が画像 → 運転手・クラスを与える。
test 画像には運転手の情報が無い。
"""

from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
OUTPUT_DIR = COMPETITION_DIR / "outputs"
IMG_DIR = DATA_DIR / "imgs"
DRIVERS_CSV = DATA_DIR / "driver_imgs_list.csv"
SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

CLASSES = tuple(f"c{i}" for i in range(10))
N_CLASSES = len(CLASSES)
DRIVER = "subject"
IMG = "img"


def load_train() -> pd.DataFrame:
    """1 行 = 1 枚の学習画像。列は subject・classname・img・label（0〜9）・path。運転手・画像名の順。"""
    df = pd.read_csv(DRIVERS_CSV)
    df = df.sort_values([DRIVER, IMG], kind="stable").reset_index(drop=True)
    return df.assign(
        label=df["classname"].map({c: i for i, c in enumerate(CLASSES)}).astype(np.int64),
        path=[
            str(IMG_DIR / "train" / c / i) for c, i in zip(df["classname"], df[IMG], strict=True)
        ],
    )


def load_test() -> pd.DataFrame:
    """提出ファイルと同じ行順の test 画像。列は img・path。"""
    df = pd.read_csv(SUBMISSION_CSV, usecols=[IMG])
    return df.assign(path=[str(IMG_DIR / "test" / i) for i in df[IMG]])


def submission(test: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """test の行順のまま、確率を提出形式（img + c0〜c9）にする。"""
    return pd.concat(
        [test[[IMG]].reset_index(drop=True), pd.DataFrame(proba, columns=list(CLASSES))], axis=1
    )
