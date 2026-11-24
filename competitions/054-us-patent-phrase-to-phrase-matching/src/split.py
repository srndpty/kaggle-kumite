"""正規化 anchor と無順序ペアで連結したグループを分割する。"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def groups(rows: pd.DataFrame) -> np.ndarray:
    parents = {anchor: anchor for anchor in rows["anchor"]}

    def root(anchor: str) -> str:
        while anchor != parents[anchor]:
            parents[anchor] = parents[parents[anchor]]
            anchor = parents[anchor]
        return anchor

    owners: dict[tuple[str, ...], str] = {}
    for anchor, target in rows[["anchor", "target"]].itertuples(index=False, name=None):
        pair = tuple(sorted((anchor, target)))
        other = owners.setdefault(pair, anchor)
        left, right = root(anchor), root(other)
        parents[max(left, right)] = min(left, right)
    return np.asarray([root(anchor) for anchor in rows["anchor"]])


def make_split(rows: pd.DataFrame) -> pd.DataFrame:
    component = groups(rows)
    folds = np.full(len(rows), -1, dtype=int)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(rows, groups=component)):
        if (folds[valid] != -1).any():
            raise ValueError("検証行が重複しています")
        folds[valid] = fold
    if (folds < 0).any():
        raise ValueError("検証されていない行があります")
    return pd.DataFrame({"id": rows["id"], "component": component, "fold": folds})
