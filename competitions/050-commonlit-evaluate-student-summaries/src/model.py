"""要約評価のデータ、特徴量、回帰と提出形式。"""

import re
from collections.abc import Iterable, Set
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from kumite.text_features import build_features

TARGETS = ["content", "wording"]
FILES = (
    "prompts_train.csv",
    "prompts_test.csv",
    "summaries_train.csv",
    "summaries_test.csv",
    "sample_submission.csv",
)
WORDS = re.compile(r"\b\w+\b")
CHANGES = {
    "E00": "要約の語数、文数、平均語長、平均文長",
    "E01": "要約の単語 unigram TF-IDF を追加",
    "E02": "原文との unigram・bigram 一致率を追加",
    "E03": "設問との単語一致率を追加",
    "E04": "課題単位から行単位ランダム分割に変更",
    "E05": "原文に対する要約の語数比を追加",
}


def load_partition(data: Path, partition: str) -> pd.DataFrame:
    prompts = pd.read_csv(data / f"prompts_{partition}.csv", dtype={"prompt_id": str})
    rows = pd.read_csv(
        data / f"summaries_{partition}.csv", dtype={"student_id": str, "prompt_id": str}
    )
    if rows.empty or rows["student_id"].isna().any() or not rows["student_id"].is_unique:
        raise ValueError(f"{partition} の student_id が不正です")
    if prompts["prompt_id"].isna().any() or rows["prompt_id"].isna().any():
        raise ValueError(f"{partition} の prompt_id に欠損があります")
    joined = rows.merge(prompts, on="prompt_id", how="left", validate="many_to_one")
    columns = ["text", "prompt_text", "prompt_question"]
    if joined[columns].isna().any().any():
        raise ValueError(f"{partition} の要約・原文・設問に欠損があります")
    return joined


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_partition(data, "train")
    test = load_partition(data, "test")
    if set(train["prompt_id"]) & set(test["prompt_id"]):
        raise ValueError("学習とテストで課題が重複しています")
    if not np.isfinite(train[TARGETS].to_numpy()).all():
        raise ValueError("学習目的値に非有限値があります")
    return train, test


def tokens(text: str) -> list[str]:
    return WORDS.findall(text.casefold())


Item = TypeVar("Item")


def overlap(items: Iterable[str | tuple[str, str]], reference: Set[str | tuple[str, str]]) -> float:
    values = list(items)
    return sum(value in reference for value in values) / max(len(values), 1)


def dense_features(rows: pd.DataFrame, trial: str) -> np.ndarray:
    values = []
    for text, source, question in rows[["text", "prompt_text", "prompt_question"]].itertuples(
        index=False, name=None
    ):
        words = tokens(text)
        sentences = [part for part in re.split(r"[.!?]+", text) if WORDS.search(part)]
        features = [
            len(words),
            len(sentences),
            sum(map(len, words)) / max(len(words), 1),
            len(words) / max(len(sentences), 1),
        ]
        if trial == "E02":
            source_words = tokens(source)
            features.extend(
                (
                    overlap(words, set(source_words)),
                    overlap(pairwise(words), set(pairwise(source_words))),
                )
            )
        if trial == "E03":
            features.append(overlap(words, set(tokens(question))))
        if trial == "E05":
            features.append(len(words) / max(len(tokens(source)), 1))
        values.append(features)
    return np.asarray(values, dtype=np.float64)


def predict(train: pd.DataFrame, query: pd.DataFrame, trial: str) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(dense_features(train, trial))
    x_query = scaler.transform(dense_features(query, trial))
    if trial == "E01":
        (train_text, query_text, _), _ = build_features(
            train["text"].tolist(), query["text"].tolist(), query["text"].iloc[:1].tolist()
        )
        x_train = sparse.hstack([x_train, train_text], format="csr")
        x_query = sparse.hstack([x_query, query_text], format="csr")
    y = train[TARGETS].to_numpy()
    return np.column_stack(
        [Ridge(alpha=1.0, solver="lsqr").fit(x_train, target).predict(x_query) for target in y.T]
    )


def write_submission(
    test: pd.DataFrame, prediction: np.ndarray, output: Path, sample: Path
) -> None:
    if prediction.shape != (len(test), len(TARGETS)) or not np.isfinite(prediction).all():
        raise ValueError("提出予測の形状または値が不正です")
    submission = pd.DataFrame(prediction, columns=TARGETS, index=test["student_id"])
    template = pd.read_csv(sample, dtype={"student_id": str})
    if list(template) != ["student_id", *TARGETS] or not template["student_id"].is_unique:
        raise ValueError("サンプル提出の形式が不正です")
    if set(template["student_id"]) == set(submission.index):
        submission = submission.loc[template["student_id"]]
    submission.to_csv(output)
