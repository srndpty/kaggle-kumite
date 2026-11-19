"""質問順に依存しないペア特徴。"""

import re
from typing import cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"\b\w+\b")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NEGATION = re.compile(r"\b(?:no|not|never|without|neither|nor)\b|n['’]t\b")
EXPERIMENTS = ("E01", "E02", "E03", "E04", "E05")


def common_prefix_length(left: list[str], right: list[str]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


class PairFeatures:
    def __init__(self, experiment: str) -> None:
        self.experiment = experiment
        # scikit-learn は float32 を受け付けるが、型情報は float64 に限られる。
        sparse_dtype = cast("type[np.float64]", np.float32)
        self.word = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w+\b", dtype=sparse_dtype
        )
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, dtype=sparse_dtype
        )

    def fit(self, pairs: np.ndarray) -> None:
        questions = np.unique(pairs)
        self.word.fit(questions)
        if self.experiment == "E02":
            self.char.fit(questions)

    def transform(self, pairs: np.ndarray) -> np.ndarray:
        questions, indices = np.unique(pairs, return_inverse=True)
        indices = indices.reshape(-1, 2)

        def cosine(vectorizer: TfidfVectorizer) -> np.ndarray:
            vectors = csr_matrix(vectorizer.transform(questions))
            left = cast(csr_matrix, vectors[indices[:, 0]])
            right = cast(csr_matrix, vectors[indices[:, 1]])
            return np.asarray(left.multiply(right).sum(axis=1)).ravel()

        columns = [cosine(self.word)]
        extra = []
        for left, right in pairs:
            left_words, right_words = TOKEN.findall(left), TOKEN.findall(right)
            short, long = sorted((len(left_words), len(right_words)))
            row = [float(short), float(long), float(long - short)]
            if self.experiment == "E03":
                a, b = set(left_words), set(right_words)
                overlap = len(a & b)
                row.extend(
                    [
                        overlap / max(len(a | b), 1),
                        *sorted((overlap / max(len(a), 1), overlap / max(len(b), 1))),
                    ]
                )
            if self.experiment == "E04":
                row.extend(
                    [
                        float(set(NUMBER.findall(left)) != set(NUMBER.findall(right))),
                        float(bool(NEGATION.search(left)) != bool(NEGATION.search(right))),
                    ]
                )
            if self.experiment == "E05":
                row.extend(
                    [
                        common_prefix_length(left_words, right_words) / max(short, 1),
                        common_prefix_length(left_words[::-1], right_words[::-1]) / max(short, 1),
                    ]
                )
            extra.append(row)
        if self.experiment == "E02":
            columns.append(cosine(self.char))
        return np.column_stack([*columns, np.asarray(extra)])
