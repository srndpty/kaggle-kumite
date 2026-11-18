"""同一文面をまとめた多ラベル層化3分割。"""

import hashlib
import unicodedata

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def text_key(text: str) -> str:
    return hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode()
    ).hexdigest()


def make_folds(texts, y: np.ndarray, seed: int = 42, n_splits: int = 3):
    keys = np.array([text_key(t) for t in texts])
    groups, inverse = np.unique(keys, return_inverse=True)
    group_y = np.zeros((len(groups), y.shape[1]), dtype=np.int64)
    np.maximum.at(group_y, inverse, y)
    group_folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(splitter.split(np.zeros(len(groups)), group_y)):
        group_folds[va] = k
    folds = group_folds[inverse]
    for k in range(n_splits):
        for mask in (folds == k, folds != k):
            if np.any(y[mask].sum(axis=0) == 0) or np.any(y[mask].sum(axis=0) == mask.sum()):
                raise ValueError("分割内に片クラスのラベルがあります")
    return folds
