"""語彙とIDFは外側foldの学習文だけから作る。"""

from collections.abc import Iterable
from typing import Any, Protocol, cast

import numpy as np
from scipy import sparse  # pyright: ignore[reportMissingTypeStubs]
from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingTypeStubs]


class _TextTransform(Protocol):
    def fit_transform(self, documents: Iterable[str]) -> sparse.csr_matrix: ...
    def transform(self, documents: Iterable[str]) -> sparse.csr_matrix: ...


class _SparseOperations(Protocol):
    def hstack(self, blocks: list[sparse.csr_matrix], format: str) -> sparse.csr_matrix: ...


def build_features(
    train: Iterable[str],
    valid: Iterable[str],
    test: Iterable[str],
    *,
    bigram: bool = False,
    chars: bool = False,
    additional_bigrams: bool = False,
    lowercase: bool = True,
) -> tuple[tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix], list[TfidfVectorizer]]:
    settings = [("word", (1, 2 if bigram else 1), 100_000)]
    if additional_bigrams:
        settings.append(("word", (2, 2), 100_000))
    if chars:
        settings.append(("char_wb", (3, 5), 150_000))
    vectorizers = [
        TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=3,
            max_features=max_features,
            sublinear_tf=True,
            lowercase=lowercase,
            dtype=cast(Any, np.float32),
        )
        for analyzer, ngram_range, max_features in settings
    ]
    transforms = [cast(_TextTransform, vectorizer) for vectorizer in vectorizers]
    stack = cast(_SparseOperations, sparse).hstack
    training = stack([transform.fit_transform(train) for transform in transforms], "csr")
    validation = stack([transform.transform(valid) for transform in transforms], "csr")
    query = stack([transform.transform(test) for transform in transforms], "csr")
    return (training, validation, query), vectorizers
