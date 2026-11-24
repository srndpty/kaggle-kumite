"""Kaggle の信頼度順・最近傍マッチによる Event Detection AP。"""

import numpy as np
import polars as pl
from pipeline import EVENTS

TOLERANCES = (12, 36, 60, 90, 120, 150, 180, 240, 300, 360)


def confidence_order(confidence: np.ndarray) -> np.ndarray:
    """公式実装の pandas sort_values と同じ同点順。"""
    return (len(confidence) - 1 - confidence[::-1].argsort(kind="quicksort"))[::-1]


def matches(truth: np.ndarray, steps: np.ndarray, tolerance: int) -> np.ndarray:
    found = np.zeros(len(steps), dtype=bool)
    used = np.zeros(len(truth), dtype=bool)
    lower = np.searchsorted(truth, steps - tolerance + 1)
    upper = np.searchsorted(truth, steps + tolerance)
    for i in np.flatnonzero(upper > lower):
        lo, hi = lower[i], upper[i]
        step = steps[i]
        remaining = np.flatnonzero(~used[lo:hi]) + lo
        if remaining.size:
            closest = remaining[np.abs(truth[remaining] - step).argmin()]
            used[closest] = found[i] = True
    return found


def score(truth: pl.DataFrame, predicted: pl.DataFrame) -> dict[str, float]:
    valid = truth.drop_nulls("step")
    series_ids = valid["series_id"].unique().to_list()
    detail: dict[str, float] = {}
    for event in EVENTS:
        actual = valid.filter(pl.col("event") == event)
        groups = {
            key: value["step"].sort().to_numpy()
            for key, value in actual.partition_by("series_id", as_dict=True).items()
        }
        proposals = predicted.filter(
            (pl.col("event") == event) & pl.col("series_id").is_in(series_ids)
        )
        parts = []
        for key, part in proposals.partition_by("series_id", as_dict=True).items():
            ordered = part.sort("step")
            confidence = ordered["score"].to_numpy()
            order = confidence_order(confidence)
            parts.append(
                (
                    groups.get(key, np.empty(0, dtype=np.int64)),
                    ordered["step"].to_numpy()[order],
                    confidence[order],
                )
            )
        if not parts or actual.is_empty():
            detail.update({f"{event}/{t}": 0.0 for t in TOLERANCES})
            continue
        scores = np.concatenate([part[2] for part in parts])
        order = np.argsort(-scores, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
        for tolerance in TOLERANCES:
            found = np.concatenate([matches(a, p, tolerance) for a, p, _ in parts])
            tp = found[order].cumsum()[ends]
            detail[f"{event}/{tolerance}"] = float(
                np.sum(np.diff(tp, prepend=0) * tp / (ends + 1)) / len(actual)
            )
    return detail
