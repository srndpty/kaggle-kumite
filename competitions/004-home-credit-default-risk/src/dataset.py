"""公式CSVの読込と、申込者1行の不変条件。"""

from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
DATA_FILES = tuple(
    DATA_DIR / name
    for name in (
        "application_train.csv",
        "application_test.csv",
        "sample_submission.csv",
        "bureau.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "HomeCredit_columns_description.csv",
    )
)
KEY = "SK_ID_CURR"


def unique_ids(frame: pd.DataFrame, key: str = KEY) -> None:
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{key} は欠損のない一意キーが必要です")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    for frame in (train, test, sample):
        unique_ids(frame)
    if set(train[KEY]) & set(test[KEY]):
        raise ValueError("train/test の顧客が重複しています")
    if not np.isin(train["TARGET"], [0, 1]).all():
        raise ValueError("TARGET は0/1が必要です")
    if list(train.drop(columns="TARGET").columns) != list(test.columns):
        raise ValueError("train/test の列が一致しません")
    if not np.array_equal(sample[KEY], test[KEY]):
        raise ValueError("sample/test のID順序が一致しません")
    return train, test, sample


def submission(sample: pd.DataFrame, ids: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    unique_ids(sample)
    if list(sample.columns) != [KEY, "TARGET"] or not np.array_equal(sample[KEY], ids):
        raise ValueError("提出の列またはID順序が不正です")
    if pred.shape != (len(sample),) or not np.isfinite(pred).all():
        raise ValueError("提出予測の形状または有限性が不正です")
    if ((pred < 0) | (pred > 1)).any():
        raise ValueError("提出確率が範囲外です")
    return sample.assign(TARGET=pred)
