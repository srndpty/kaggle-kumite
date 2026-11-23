"""抜粋の重複と出典情報を使って CV のグループを固定する。"""

import unicodedata
from collections.abc import Iterator
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
VERSION = "commonlit-group-v1"


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str) -> Iterator[str]:
    for i in range(len(text) - 4):
        yield text[i : i + 5]


def groups_for(train: pd.DataFrame) -> np.ndarray:
    texts = [normalize(text) for text in train["excerpt"]]
    vectorizer = CountVectorizer(analyzer=cast(Any, shingles), binary=True)
    matrix = cast(csr_matrix, vectorizer.fit_transform(texts))
    counts = matrix.getnnz(axis=1)
    edges: list[tuple[int, int]] = []
    for start in range(0, len(texts), 128):
        intersections = (matrix[start : start + 128] @ matrix.T).tocoo()
        rows = intersections.row + start
        cols = intersections.col
        union = counts[rows] + counts[cols] - intersections.data
        keep = (cols < rows) & (5 * intersections.data >= 4 * union)
        edges.extend(zip(rows[keep].tolist(), cols[keep].tolist(), strict=True))
    known: dict[tuple[str, ...], int] = {}
    for i, (text, url) in enumerate(zip(texts, train["url_legal"], strict=True)):
        keys: list[tuple[str, ...]] = [("text", text)]
        if isinstance(url, str):
            parts = urlsplit(url.strip())
            if parts.hostname and parts.path.strip("/"):
                keys.append(("url", parts.netloc.casefold(), parts.path.rstrip("/"), parts.query))
        for key in keys:
            edges.append((i, known.setdefault(key, i)))
    rows, cols = zip(*edges, strict=True)
    graph = csr_matrix(
        (np.ones(len(edges)), (list(rows), list(cols))), shape=(len(texts), len(texts))
    )
    _, groups = connected_components(graph, directed=False)
    return np.asarray(groups)


def make_folds(train: pd.DataFrame) -> pd.DataFrame:
    groups = groups_for(train)
    strata = pd.qcut(train["target"], 5, labels=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(train)), strata, groups)):
        folds[valid] = fold
    result = pd.DataFrame({"id": train["id"], "group": groups, "fold": folds})
    if set(folds.tolist()) != set(range(5)) or result.groupby("group")["fold"].nunique().max() != 1:
        raise ValueError("分割の被覆またはグループ独立性が不正です")
    return result
