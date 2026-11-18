"""公式 CSV の読み込みと提出 ID の照合。"""

from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMPETITION_DIR / "data"
DATA_FILES = tuple(DATA_DIR / name for name in ("train.csv", "test.csv", "sample_submission.csv"))
TARGET = "target"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (pd.read_csv(p) for p in DATA_FILES)
    for frame in (train, test, sample):
        if not frame["id"].is_unique or frame["id"].isna().any():
            raise ValueError("ID に重複または欠損があります")
    if set(train.columns) != set(test.columns) | {TARGET}:
        raise ValueError("train/test の列が一致しません")
    if set(sample.columns) != {"id", TARGET} or set(sample["id"]) != set(test["id"]):
        raise ValueError("sample submission と test の ID が一致しません")
    if not np.isin(train[TARGET], [0, 1]).all():
        raise ValueError("ラベルは 0/1 が必要です")
    test = test.set_index("id").loc[sample["id"]].reset_index()
    return train, test, sample


def feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["id", TARGET], errors="ignore")


def submission(sample: pd.DataFrame, ids: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    if not np.array_equal(sample["id"].to_numpy(), ids) or pred.shape != (len(sample),):
        raise ValueError("提出 ID 順または予測形状が一致しません")
    from kumite.checks import validate_probabilities

    validate_probabilities(np.column_stack((1 - pred, pred)), len(pred), 2)
    return sample.assign(target=pred)
