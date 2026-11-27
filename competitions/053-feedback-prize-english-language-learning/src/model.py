"""英文エッセイの６軸を予測する CPU モデル。"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TARGETS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
SLUG = "feedback-prize-english-language-learning"


def read_essays(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"text_id": str})
    ids = frame["text_id"]
    if ids.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError(f"不正な text_id: {path}")
    return frame


def duplicate_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def length_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        sentences = re.findall(r"[^.!?]*\w[^.!?]*", text)
        rows.append(
            (
                len(words),
                len(text),
                len(sentences),
                len(words) / max(1, len(sentences)),
                len(set(words)) / max(1, len(words)),
                sum(unicodedata.category(char).startswith("P") for char in text)
                / max(1, len(text)),
            )
        )
    return np.asarray(rows, dtype=float)


def validate_labels(labels: np.ndarray, n_rows: int) -> None:
    if n_rows == 0 or labels.shape != (n_rows, len(TARGETS)):
        raise ValueError("学習ラベルは空でない６列で、本文と同じ行数が必要です")
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
        raise ValueError("学習ラベルには有限な実数が必要です")
    if not ((labels >= 1) & (labels <= 5)).all():
        raise ValueError("学習ラベルは [1, 5] の範囲が必要です")


def fit_model(texts: list[str], labels: np.ndarray, block: str):
    validate_labels(labels, len(texts))
    if block not in {"base", "bigram", "char", "length", "rbf"}:
        raise ValueError(f"未知の特徴量設定: {block}")
    vectorizers = [TfidfVectorizer(ngram_range=(1, 1), min_df=2, dtype=np.float64)]
    if block == "bigram":
        vectorizers.append(TfidfVectorizer(ngram_range=(2, 2), min_df=2, dtype=np.float64))
    if block == "char":
        vectorizers.append(
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, dtype=np.float64)
        )
    scaler = StandardScaler().fit(length_features(texts)) if block == "length" else None
    for vectorizer in vectorizers:
        vectorizer.fit(texts)

    def features(rows: list[str]):
        blocks = [vectorizer.transform(rows) for vectorizer in vectorizers]
        if scaler is not None:
            blocks.append(csr_matrix(scaler.transform(length_features(rows))))
        return hstack(blocks, format="csr")

    ridge = (
        KernelRidge(alpha=1, kernel="rbf", gamma=1.0)
        if block == "rbf"
        else Ridge(alpha=1.0, solver="lsqr")
    )
    train_features = features(texts)
    # KernelRidge にも Ridge と同じ訓練平均の切片を与える。
    offset = labels.mean(axis=0) if block == "rbf" else 0.0
    ridge.fit(train_features, labels - offset)

    def predict(rows: list[str]) -> np.ndarray:
        result = np.asarray(ridge.predict(features(rows)), dtype=float) + offset
        if result.shape != (len(rows), len(TARGETS)) or not np.isfinite(result).all():
            raise ValueError("予測の形状または値が不正です")
        return np.clip(result, 1.0, 5.0)

    return predict


def write_submission(data: Path, output: Path, block: str) -> None:
    train = read_essays(data / "train.csv")
    test = read_essays(data / "test.csv")
    submission = read_essays(data / "sample_submission.csv")
    if list(submission.columns) != ["text_id", *TARGETS]:
        raise ValueError("提出サンプルの列が不正です")
    if set(test["text_id"]) != set(submission["text_id"]):
        raise ValueError("テストと提出サンプルの ID が一致しません")
    rows = test.set_index("text_id").loc[submission["text_id"], "full_text"]
    predict = fit_model(
        train["full_text"].fillna("").tolist(), train[list(TARGETS)].to_numpy(float), block
    )
    submission[list(TARGETS)] = predict(rows.fillna("").tolist())
    submission.to_csv(output, index=False)
