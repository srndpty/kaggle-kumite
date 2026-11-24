"""質問文の分割、学習、閾値選定。"""

import unicodedata

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from kumite.text_features import build_features

SEED = 42


def group_keys(text: pd.Series) -> np.ndarray:
    return np.array(
        [" ".join(unicodedata.normalize("NFKC", value).casefold().split()) for value in text],
        dtype=object,
    )


def splits(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    codes, unique = pd.factorize(groups)
    strata = np.zeros(len(unique), dtype=np.int8)
    np.maximum.at(strata, codes, y)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = np.empty(len(unique), dtype=np.int8)
    for k, (_, valid) in enumerate(splitter.split(strata, strata)):
        folds[valid] = k
    assignments = folds[codes]
    return [(np.flatnonzero(assignments != k), np.flatnonzero(assignments == k)) for k in range(3)]


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if not len(p) or not np.any(y == 1):
        raise ValueError("閾値選定には陽性例が必要です")
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    endpoints = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    f1 = 2 * np.cumsum(y[order])[endpoints] / (endpoints + 1 + y.sum())
    return float(scores[endpoints[np.argmax(f1)]])


def predict(
    text: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    tune: np.ndarray,
    query: np.ndarray,
    *,
    chars: bool = False,
    balanced: bool = False,
    preserve_case: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (x_fit, x_tune, x_query), vectorizers = build_features(
        text[fit], text[tune], query, chars=chars, lowercase=not preserve_case
    )
    model = LogisticRegression(
        C=1, class_weight="balanced" if balanced else None, max_iter=200, random_state=SEED
    ).fit(x_fit, y[fit])
    vocabulary = vectorizers[0].vocabulary_
    tokenize = vectorizers[0].build_analyzer()

    def coverage(value: str) -> float:
        tokens = tokenize(value)
        return sum(token in vocabulary for token in tokens) / len(tokens) if tokens else 0.0

    return (
        model.predict_proba(x_tune)[:, 1],
        model.predict_proba(x_query)[:, 1],
        np.fromiter(map(coverage, query), dtype=np.float32, count=len(query)),
    )
