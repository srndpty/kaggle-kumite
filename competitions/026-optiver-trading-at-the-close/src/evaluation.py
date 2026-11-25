"""取引日順の評価対象と分割を作る。"""

import numpy as np

from kumite.crossfit import Split


def splits_for(days: np.ndarray, y: np.ndarray) -> list[Split]:
    ordered = np.unique(days)
    if len(ordered) <= 90:
        raise ValueError("At least 91 trading days are required")
    scored = np.isfinite(y)
    splits = []
    for window in ordered[-90:].reshape(3, 30):
        first, last = window[0], window[-1]
        splits.append(
            (
                np.flatnonzero(scored & (days < first)),
                np.flatnonzero(scored & (days >= first) & (days <= last)),
            )
        )
    return splits


def mae(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(y - predictions)))
