"""重複グループを保った多ラベル層化。"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def multilabel_group_folds(
    keys: Sequence[str], labels: npt.NDArray[np.bool_], *, n_folds: int, seed: int
) -> npt.NDArray[np.int64]:
    if labels.ndim != 2 or labels.shape[0] != len(keys) or n_folds < 2:
        raise ValueError("Invalid multilabel group split inputs")
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    if len(groups) < n_folds:
        raise ValueError("Too few independent curated groups")
    rng = np.random.default_rng(seed)
    group_rows = list(groups.values())
    members: list[list[int]] = [group_rows[int(i)] for i in rng.permutation(len(groups))]
    totals = labels.sum(axis=0).clip(min=1)
    counts = [labels[group].sum(axis=0) for group in members]
    order = sorted(range(len(members)), key=lambda i: -float((counts[i] / totals).sum()))
    counts_per_fold = np.zeros((n_folds, labels.shape[1]))
    sizes = np.zeros(n_folds)
    folds = np.empty(len(keys), dtype=np.int64)
    for i in order:
        count = counts[i]
        group = members[i]
        # タグごとの正例数とクリップ総数の二乗偏差の増分を最小化する。
        costs = ((2 * counts_per_fold + count) * count / totals**2).sum(axis=1)
        costs += (2 * sizes + len(group)) * len(group) / len(keys) ** 2
        fold = int(np.argmin(costs))
        counts_per_fold[fold] += count
        sizes[fold] += len(group)
        folds[group] = fold
    if not sizes.all():
        raise ValueError("Empty validation fold")
    return folds
