"""未知の質問ペアを評価するための分割。"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from kumite.crossfit import Split, split_roles


def question_nodes(frame: pd.DataFrame) -> np.ndarray:
    ids = frame[["qid1", "qid2"]].to_numpy().ravel()
    text = pd.Series(frame[["question1", "question2"]].to_numpy().ravel(), dtype="string")
    text = text.fillna("").str.normalize("NFKC").str.casefold().str.split().str.join(" ")
    qids, unique_ids = pd.factorize(ids)
    texts, unique_texts = pd.factorize(text)
    known = text.to_numpy() != ""
    size = len(unique_ids) + len(unique_texts)
    graph = coo_matrix(
        (np.ones(known.sum()), (qids[known], len(unique_ids) + texts[known])),
        shape=(size, size),
    )
    _, labels = connected_components(graph.tocsr(), directed=False)
    return labels[qids].reshape(-1, 2)


def components(nodes: np.ndarray) -> np.ndarray:
    size = int(nodes.max()) + 1
    graph = coo_matrix((np.ones(len(nodes)), (nodes[:, 0], nodes[:, 1])), shape=(size, size))
    _, labels = connected_components(graph.tocsr(), directed=False)
    return labels[nodes[:, 0]]


def node_splits(nodes: np.ndarray, seed: int) -> list[Split]:
    assignments = np.random.default_rng(seed).integers(0, 5, size=int(nodes.max()) + 1)
    sides = assignments[nodes]
    return [
        (np.flatnonzero((sides != k).all(axis=1)), np.flatnonzero((sides == k).all(axis=1)))
        for k in range(5)
    ]


def make_splits(nodes: np.ndarray, y: np.ndarray, seed: int) -> tuple[str, list[Split]]:
    groups = components(nodes)

    def has_classes(rows: np.ndarray) -> bool:
        return len(np.unique(y[rows])) == 2

    splits = []
    _, counts = np.unique(groups, return_counts=True)
    if len(counts) >= 5 and counts.max() <= 0.3 * len(y) and np.bincount(y).min() >= 5:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        splits = list(splitter.split(y, y, groups))
    if splits and all(
        0.1 <= len(valid) / len(y) <= 0.3 and has_classes(train) and has_classes(valid)
        for train, valid in splits
    ):
        version = "components-v1"
    else:
        version = "nodes-v1"
        splits = node_splits(nodes, seed)
    split_roles(len(y), splits)
    for train, valid in splits:
        if not has_classes(train) or not has_classes(valid):
            raise ValueError("学習または検証に両クラスがありません")
        if np.intersect1d(nodes[train], nodes[valid]).size:
            raise ValueError("学習と検証が質問を共有しています")
    return f"{version}-seed{seed}", splits
