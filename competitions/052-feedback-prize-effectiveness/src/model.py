"""論証要素３クラスの分類と Kaggle 提出生成。"""

import hashlib
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

SLUG = "feedback-prize-effectiveness"
CLASSES = ("Ineffective", "Adequate", "Effective")
ID = "discourse_id"
ESSAY = "essay_id"
TEXT = "discourse_text"
ROLE = "discourse_type"
TARGET = "discourse_effectiveness"


def read_discourse(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = [ID, ESSAY, TEXT, ROLE]
    if any(column not in frame or frame[column].str.strip().eq("").any() for column in columns):
        raise ValueError(f"論証データに欠損があります: {path}")
    if frame.empty or not frame[ID].is_unique:
        raise ValueError(f"論証 ID が不正です: {path}")
    return frame


def labels_for(frame: pd.DataFrame) -> np.ndarray:
    encoded = frame[TARGET].map({label: i for i, label in enumerate(CLASSES)})
    if encoded.isna().any():
        raise ValueError("未知の論証ラベルがあります")
    return encoded.to_numpy(dtype=np.int64)


def other_discourses(frame: pd.DataFrame) -> list[str]:
    texts: dict[str, str] = {}
    for _, members in frame.groupby(ESSAY, sort=False):
        pairs = list(members[[ID, TEXT]].itertuples(index=False, name=None))
        for key, _ in pairs:
            texts[key] = "\n".join(text for other, text in pairs if other != key)
    return [texts[key] for key in frame[ID]]


def check_probabilities(values: np.ndarray, count: int) -> None:
    if (
        values.shape != (count, len(CLASSES))
        or not (np.isfinite(values) & (values >= 0) & (values <= 1)).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("３クラスの確率が不正です")


def boundary_text(text: str) -> str:
    words = re.findall(r"\b\w\w+\b", text.lower())
    return " ".join(
        [*(f"start_{word}" for word in words[:8]), *(f"end_{word}" for word in words[-8:])]
    )


class FitResult(NamedTuple):
    predict: Callable[[pd.DataFrame], np.ndarray]
    n_iter: list[int]
    convergence_warning: bool


def fit_model(train: pd.DataFrame, block: str) -> FitResult:
    if block not in {"base", "role", "char", "context", "boundary"}:
        raise ValueError(f"未知の特徴量構成: {block}")
    labels = labels_for(train)
    if set(labels) != set(range(len(CLASSES))):
        raise ValueError("学習 fold に全３クラスが必要です")
    word = TfidfVectorizer(min_df=2, dtype=np.float64).fit(train[TEXT])
    role = OneHotEncoder(handle_unknown="ignore").fit(train[[ROLE]]) if block == "role" else None
    char = (
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, dtype=np.float64).fit(
            train[TEXT]
        )
        if block == "char"
        else None
    )
    context = (
        TfidfVectorizer(min_df=2, dtype=np.float64).fit(other_discourses(train))
        if block == "context"
        else None
    )
    boundary = (
        TfidfVectorizer(min_df=2, dtype=np.float64).fit(train[TEXT].map(boundary_text))
        if block == "boundary"
        else None
    )

    def features(frame: pd.DataFrame):
        matrices = [word.transform(frame[TEXT])]
        if role is not None:
            matrices.append(role.transform(frame[[ROLE]]))
        if char is not None:
            matrices.append(char.transform(frame[TEXT]))
        if context is not None:
            matrices.append(context.transform(other_discourses(frame)))
        if boundary is not None:
            matrices.append(boundary.transform(frame[TEXT].map(boundary_text)))
        return hstack(matrices, format="csr")

    model = LogisticRegression(C=1, solver="lbfgs", max_iter=1000, tol=1e-4)
    train_features = features(train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_features, labels)
    for warning in caught:
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(model.predict_proba(features(frame)), dtype=np.float64)
        check_probabilities(values, len(frame))
        return values

    return FitResult(
        predict,
        model.n_iter_.tolist(),
        any(issubclass(w.category, ConvergenceWarning) for w in caught),
    )


def write_submission(data: Path, output: Path, block: str, expected_train_sha256: str) -> None:
    train_path = data / "train.csv"
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != expected_train_sha256:
        raise ValueError("Training data hash mismatch")
    train = read_discourse(train_path)
    test = read_discourse(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv", dtype={ID: str}, keep_default_na=False)
    if list(sample.columns) != [ID, *CLASSES] or not sample[ID].is_unique:
        raise ValueError("提出サンプルの列または ID が不正です")
    if set(sample[ID]) != set(test[ID]):
        raise ValueError("提出サンプルとテストの ID が一致しません")
    test = test.set_index(ID).loc[sample[ID]].reset_index()
    fitted = fit_model(train, block)
    print(f"Optimization: n_iter={fitted.n_iter}, convergence_warning={fitted.convergence_warning}")
    sample[list(CLASSES)] = fitted.predict(test)
    sample.to_csv(output, index=False)
