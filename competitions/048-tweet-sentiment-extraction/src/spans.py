"""感情を根拠づける原文の連続区間を抽出する。"""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ID = "textID"
TARGET = "selected_text"
SEED = 42
FILES = ("train.csv", "test.csv", "sample_submission.csv")


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (
        pd.read_csv(data / name, dtype=str, keep_default_na=False) for name in FILES
    )
    usable = train["text"].str.strip().ne("") & train[TARGET].str.strip().ne("")
    return train.loc[usable].reset_index(drop=True), test, sample


def training_sha256(train: pd.DataFrame) -> str:
    """読み込み後の学習行を、文字列を保持した UTF-8 JSON Lines として識別する。"""
    digest = hashlib.sha256()
    for row in train[[ID, "text", "sentiment", TARGET]].itertuples(index=False, name=None):
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        digest.update(encoded.encode("utf-8") + b"\n")
    return digest.hexdigest()


def group_key(text: str) -> str:
    return " ".join(text.casefold().split())


def make_folds(train: pd.DataFrame) -> np.ndarray:
    groups = train["text"].map(group_key)
    folds = np.full(len(train), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(train, train["sentiment"], groups)):
        folds[valid] = fold
    return folds


def words(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\S+", text))


def position(text: str, answer: str) -> tuple[int, int] | None:
    answer = answer.strip()
    start = text.find(answer)
    if not answer or start < 0 or text.find(answer, start + 1) >= 0:
        return None
    return start, start + len(answer)


def jaccard(answer: str, prediction: str) -> float:
    a, b = set(answer.lower().split()), set(prediction.lower().split())
    return len(a & b) / len(a | b)


def features(text: str, sentiment: str, trial: str) -> list[dict[str, float]]:
    tokens = [match.group().lower() for match in words(text)]
    result = []
    for i, token in enumerate(tokens):
        row = {
            f"sentiment={sentiment}": 1.0,
            f"word={token}": 1.0,
            f"word={sentiment}:{token}": 1.0,
        }
        if trial == "E02":
            for n in (3, 4, 5):
                for j in range(len(token) - n + 1):
                    row[f"char={token[j : j + n]}"] = 1.0
        if trial == "E03":
            if i:
                row[f"prev={tokens[i - 1]}"] = 1.0
            if i + 1 < len(tokens):
                row[f"next={tokens[i + 1]}"] = 1.0
        result.append(row)
    return result


def fit(train: pd.DataFrame, trial: str) -> tuple[DictVectorizer, LogisticRegression]:
    rows = []
    labels: list[int] = []
    for text, sentiment, answer in train[["text", "sentiment", TARGET]].itertuples(
        index=False, name=None
    ):
        interval = position(text, answer)
        if interval is None:
            continue
        start, end = interval
        rows.extend(features(text, sentiment, trial))
        labels.extend(int(match.start() < end and start < match.end()) for match in words(text))
    vectorizer = DictVectorizer()
    model = LogisticRegression(solver="liblinear", max_iter=200, random_state=SEED)
    matrix = vectorizer.fit_transform(rows)
    # liblinear は疎行列のインデックスに int32 を要求する。
    if matrix.nnz > np.iinfo(np.int32).max:
        raise ValueError("liblinear の疎行列サイズ上限を超えています")
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    model.fit(matrix, labels)
    return vectorizer, model


def score_tokens(
    frame: pd.DataFrame, trained: tuple[DictVectorizer, LogisticRegression], trial: str
) -> np.ndarray:
    vectorizer, model = trained
    rows = [
        row
        for text, sentiment in frame[["text", "sentiment"]].itertuples(index=False, name=None)
        for row in features(text, sentiment, trial)
    ]
    probability = model.predict_proba(vectorizer.transform(rows))
    if (
        probability.shape != (len(rows), 2)
        or not np.isfinite(probability).all()
        or not ((probability >= 0) & (probability <= 1)).all()
        or not np.allclose(probability.sum(axis=1), 1)
    ):
        raise ValueError("語の確率が不正です")
    return 2 * probability[:, 1] - 1


def best_span(scores: np.ndarray) -> tuple[int, int]:
    best = -np.inf
    current = 0.0
    start = 0
    span = (0, 0)
    for i, score in enumerate(scores):
        if current < 0:
            current = 0.0
            start = i
        current += score
        if current > best:
            best = current
            span = (start, i)
    return span


def jaccard_span(tokens: list[re.Match[str]], scores: np.ndarray) -> tuple[int, int]:
    # 同じ語は集合で1回だけ数え、出現位置の最大確率で正解への包含を近似する。
    terms = [token.group().lower() for token in tokens]
    probabilities: dict[str, float] = {}
    for term, score in zip(terms, scores, strict=True):
        probabilities[term] = max(probabilities.get(term, 0.0), float((score + 1) / 2))
    total = sum(probabilities.values())
    best = -1.0
    span = (0, 0)
    for start in range(len(terms)):
        seen: set[str] = set()
        intersection = 0.0
        for end in range(start, len(terms)):
            term = terms[end]
            if term in seen:
                continue
            seen.add(term)
            intersection += probabilities[term]
            value = intersection / (len(seen) + total - intersection)
            if value > best:
                best = value
                span = (start, end)
    return span


def decode(frame: pd.DataFrame, scores: np.ndarray, trial: str) -> list[str]:
    result = []
    offset = 0
    for text, sentiment in frame[["text", "sentiment"]].itertuples(index=False, name=None):
        tokens = words(text)
        row = scores[offset : offset + len(tokens)]
        offset += len(tokens)
        if len(row) != len(tokens) or not tokens:
            raise ValueError("本文と語スコアの形状が一致しません")
        if trial == "E04" and sentiment == "neutral":
            result.append(text)
            continue
        start, end = jaccard_span(tokens, row) if trial == "E05" else best_span(row)
        result.append(text[tokens[start].start() : tokens[end].end()])
    if offset != len(scores):
        raise ValueError("余分な語スコアがあります")
    return result


def write_submission(
    test: pd.DataFrame, sample: pd.DataFrame, predictions: list[str], path: Path
) -> None:
    if any(
        not pred.strip() or pred not in text
        for text, pred in zip(test["text"], predictions, strict=True)
    ):
        raise ValueError("空欄または原文に存在しない予測があります")
    by_id = pd.Series(predictions, index=test[ID])
    if not by_id.index.is_unique or not sample[ID].is_unique or set(sample[ID]) != set(by_id.index):
        raise ValueError("提出 ID が一致しません")
    sample.assign(selected_text=sample[ID].map(by_id))[[ID, TARGET]].to_csv(
        path, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
