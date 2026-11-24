"""公式 CSV の検証、語句の正規化、予測の保存。"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from kumite.checks import validate_regression

INPUT_COLUMNS = ["id", "anchor", "target", "context"]


def normalize(phrase: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", phrase).lower().split())


def read_rows(path: Path, *, labeled: bool = False) -> pd.DataFrame:
    columns = [*INPUT_COLUMNS, "score"] if labeled else INPUT_COLUMNS
    frame = pd.read_csv(path, dtype=dict.fromkeys(INPUT_COLUMNS, str), keep_default_na=False)
    if list(frame.columns) != columns or frame.empty or not frame["id"].is_unique:
        raise ValueError(f"列または ID が不正です: {path}")
    for column in INPUT_COLUMNS:
        if not frame[column].str.strip().ne("").all():
            raise ValueError(f"空の入力があります: {path} / {column}")
    if not frame["context"].str.fullmatch(r"[A-HY][0-9]{2}").all():
        raise ValueError(f"CPC コードが不正です: {path}")
    if labeled:
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
        if not frame["score"].between(0, 1).all():
            raise ValueError(f"スコアの範囲が不正です: {path}")
    for column in ("anchor", "target"):
        frame[column] = frame[column].map(normalize)
    return frame


def load_data(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = read_rows(data / "train.csv", labeled=True)
    test = read_rows(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})
    if list(sample.columns) != ["id", "score"] or sample["id"].tolist() != test["id"].tolist():
        raise ValueError("サンプル提出と公式テストの ID 順が一致しません")
    # 公開テストは学習行の抜粋。隠しテストの採点は Kaggle 上で行う。
    return train, test


def submission(test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    validate_regression(prediction, len(test), lower=0)
    if (prediction > 1).any():
        raise ValueError("予測が 1 を超えています")
    return pd.DataFrame({"id": test["id"].to_numpy(), "score": prediction})
