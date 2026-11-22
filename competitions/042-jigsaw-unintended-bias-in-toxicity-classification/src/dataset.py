"""公式 CSV の読み込み、分割、提出 ID の検証。"""

from pathlib import Path

import numpy as np
import pandas as pd
from metrics import IDENTITIES, check_support
from sklearn.model_selection import StratifiedGroupKFold

FILES = ("train.csv", "test.csv", "sample_submission.csv")
SEED = 42
SPLIT_VERSION = "normalized-text-group-3fold-v1-seed42"


def load_train(data: Path) -> pd.DataFrame:
    train = pd.read_csv(
        data / FILES[0],
        usecols=["id", "comment_text", "target", *IDENTITIES],
        dtype={column: "float32" for column in ("target", *IDENTITIES)},
        keep_default_na=False,
        na_values={column: [""] for column in ("target", *IDENTITIES)},
    )
    if not train["target"].between(0, 1).all() or not train["id"].is_unique:
        raise ValueError("学習 ID の重複または target の欠損・範囲外の値があります")
    return train


def load_query(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(data / FILES[1], usecols=["id", "comment_text"], keep_default_na=False)
    sample = pd.read_csv(data / FILES[2])
    if list(sample.columns) != ["id", "prediction"]:
        raise ValueError("公式サンプルの提出列が想定と異なります")
    if not (
        test["id"].is_unique
        and sample["id"].is_unique
        and len(test) == len(sample)
        and sample["id"].isin(test["id"]).all()
    ):
        raise ValueError("テスト ID と提出サンプルの ID が一致しません")
    return test, sample


def labels_and_groups(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = train[["target", *IDENTITIES]].to_numpy() >= 0.5
    normalized = train["comment_text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    groups, _ = pd.factorize(normalized)
    return labels, np.asarray(groups)


def make_folds(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(labels), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(groups, labels[:, 0], groups)):
        folds[valid] = fold
    validate_folds(folds, labels, groups)
    return folds


def validate_folds(folds: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> None:
    if folds.shape != groups.shape or set(np.unique(folds).tolist()) != {0, 1, 2}:
        raise ValueError("fold に未割り当ての行または範囲外の値があります")
    order = np.argsort(groups)
    duplicate = groups[order][1:] == groups[order][:-1]
    if (np.diff(folds[order])[duplicate] != 0).any():
        raise ValueError("fold をまたぐ同一本文があります")
    for fold in range(3):
        check_support(labels[folds == fold])


def submission(test: pd.DataFrame, sample: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    validate(prediction, len(test))
    by_id = pd.Series(prediction, index=pd.Index(test["id"]))
    return sample.assign(prediction=by_id.loc[sample["id"]].to_numpy())


def validate(prediction: np.ndarray, size: int) -> None:
    if prediction.shape != (size,) or not np.isfinite(prediction).all():
        raise ValueError("予測の形状または有限性が不正です")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("予測が [0, 1] の範囲外です")
