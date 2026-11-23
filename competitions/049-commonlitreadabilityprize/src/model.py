"""公式データと提出形式を検査し、抜粋から Ridge 回帰で予測する。"""

import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from kumite.text_features import build_features

FILES = ("train.csv", "test.csv", "sample_submission.csv")
HYPOTHESES = {
    "E00": "学習ラベルの平均",
    "E01": "単語 unigram TF-IDF",
    "E02": "語内文字 3–5-gram を追加",
    "E03": "独立した単語 bigram TF-IDF を追加",
    "E04": "文章構造の数値特徴を追加",
    "E05": "回帰器を線形 SVR に変更",
}


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (pd.read_csv(data / name, dtype={"id": str}) for name in FILES)
    for frame in (train, test, sample):
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError("ID に欠損または重複があります")
    for frame in (train, test):
        if frame["excerpt"].isna().any() or frame["excerpt"].str.strip().eq("").any():
            raise ValueError("空の抜粋があります")
    if not np.isfinite(train["target"].to_numpy(dtype=float)).all():
        raise ValueError("目的値に NaN または Inf があります")
    return train, test, sample


def structure(texts: list[str]) -> np.ndarray:
    rows: list[tuple[int, int, float, float]] = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text)
        sentences = [part for part in re.split(r"[.!?]+", text) if re.search(r"\w", part)]
        rows.append(
            (
                len(words),
                len(sentences),
                sum(map(len, words)) / max(1, len(words)),
                len(words) / max(1, len(sentences)),
            )
        )
    return np.asarray(rows, dtype=float)


def predict(
    train: list[str], target: np.ndarray, query: list[str], trial: str
) -> tuple[np.ndarray, np.ndarray]:
    """予測と、各抜粋の unigram 語彙カバー率を返す。"""
    if trial == "E00":
        return np.full(len(query), target.mean()), np.zeros(len(query))
    (x_train, _, x_query), vectorizers = build_features(
        train, query, query, chars=trial == "E02", additional_bigrams=trial == "E03"
    )
    unigram = vectorizers[0]
    analyze = unigram.build_analyzer()
    coverage = []
    for text in query:
        tokens = analyze(text)
        coverage.append(sum(token in unigram.vocabulary_ for token in tokens) / max(1, len(tokens)))
    if trial == "E04":
        scaler = StandardScaler()
        x_train = hstack(
            [x_train, csr_matrix(scaler.fit_transform(structure(train)))], format="csr"
        )
        x_query = hstack([x_query, csr_matrix(scaler.transform(structure(query)))], format="csr")
    estimator = (
        LinearSVR(C=1.0, epsilon=0.1, dual=cast(Any, True), max_iter=10000, random_state=42)
        if trial == "E05"
        else Ridge(alpha=1.0, solver="lsqr")
    )
    model = estimator.fit(x_train, target)
    return np.asarray(model.predict(x_query), dtype=float), np.asarray(coverage)


def write_submission(
    test: pd.DataFrame, sample: pd.DataFrame, predictions: np.ndarray, output: Path
) -> None:
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError("提出予測の形状または有限性が不正です")
    if list(sample.columns) != ["id", "target"] or set(sample["id"]) != set(test["id"]):
        raise ValueError("提出雛形の列または ID が不正です")
    targets = pd.Series(predictions, index=test["id"])
    sample.assign(target=sample["id"].map(targets)).to_csv(output, index=False)
