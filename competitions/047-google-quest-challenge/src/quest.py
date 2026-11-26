"""Google QUEST の学習・採点と提出 CSV の検証。"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

FIELDS = ["question_title", "question_body", "answer"]
PARAMS = {"alpha": 10, "max_features": 50_000, "pca_components": 15, "quantile_max_values": 5}


def ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, axis=0, method="average")


def correlations(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    a, b = ranks(y), ranks(pred)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    norm = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide((a * b).sum(axis=0), norm, out=np.zeros_like(norm), where=norm > 0)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(correlations(y, pred).mean())


def predict(
    train: pd.DataFrame, query: pd.DataFrame, y: np.ndarray, variant: str
) -> tuple[np.ndarray, float]:
    train_text = train[FIELDS].fillna("")
    query_text = query[FIELDS].fillna("")
    if variant in {"E01", "E04"}:
        views = [(train_text[name], query_text[name], "word") for name in FIELDS]
    else:
        left = train_text.agg(" ".join, axis=1)
        right = query_text.agg(" ".join, axis=1)
        views = [(left, right, "word")]
        if variant == "E02":
            views.append((left, right, "char"))
    train_features = []
    query_features = []
    for left, right, analyzer in views:
        encoder = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(1, 2) if analyzer == "word" else (3, 5),
            max_features=PARAMS["max_features"],
        )
        train_features.append(encoder.fit_transform(left))
        query_features.append(encoder.transform(right))
    pca = PCA(n_components=min(PARAMS["pca_components"], *y.shape), svd_solver="full")
    targets = pca.fit_transform(y) if variant == "E03" else y
    model = Ridge(alpha=PARAMS["alpha"], solver="lsqr")
    model.fit(hstack(train_features, format="csr"), targets)
    raw = model.predict(hstack(query_features, format="csr"))
    if variant == "E03":
        raw = pca.inverse_transform(raw)
    clip_rate = float(((raw < 0) | (raw > 1)).mean())
    clipped = np.clip(raw, 0, 1)
    if variant == "E04":
        clipped = quantile_align(y, clipped)
    return clipped, clip_rate


def quantile_align(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    aligned = pred.copy()
    for column in range(y.shape[1]):
        if len(np.unique(y[:, column])) <= PARAMS["quantile_max_values"]:
            percentiles = (rankdata(pred[:, column]) - 0.5) / len(pred)
            aligned[:, column] = np.quantile(y[:, column], percentiles, method="inverted_cdf")
    return aligned


def submission(sample: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    columns = sample.columns.tolist()
    ids = sample["qa_id"]
    test_ids = test["qa_id"]
    if len(columns) != 31 or columns[0] != "qa_id" or len(set(columns)) != 31:
        raise ValueError("Invalid submission columns")
    if ids.isna().any() or ids.duplicated().any() or test_ids.duplicated().any():
        raise ValueError("Invalid submission IDs")
    if set(ids) != set(test_ids):
        raise ValueError("Submission and test IDs differ")
    if pred.shape != (len(test), 30) or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction shape or nonfinite values")
    if (pred < 0).any() or (pred > 1).any():
        raise ValueError("Predictions outside [0, 1]")
    result = pd.DataFrame(pred, index=test_ids, columns=columns[1:]).loc[ids]
    return result.reset_index()


def read_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(directory / "train.csv"),
        pd.read_csv(directory / "test.csv"),
        pd.read_csv(directory / "sample_submission.csv"),
    )
