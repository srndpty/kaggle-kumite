"""ラベル付きクリップの label-weighted label-ranking average precision。"""

import numpy as np
import numpy.typing as npt


def positive_precisions(
    labels: npt.NDArray[np.bool_], scores: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """正解タグの順位時点の適合率を元のタグ順で返す。"""
    if labels.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must have the same two-dimensional shape")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    # 公式参照実装と同じ降順ソートで同点も処理する。
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked = np.take_along_axis(labels, order, axis=1)
    precision = np.cumsum(ranked, axis=1) / np.arange(1, labels.shape[1] + 1)
    values = np.empty_like(scores)
    np.put_along_axis(values, order, precision * ranked, axis=1)
    return values


def lwlrap(labels: npt.NDArray[np.bool_], scores: npt.NDArray[np.float64]) -> float:
    """全正例の順位時点の適合率を平均する。"""
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("At least one positive label is required")
    return float(positive_precisions(labels, scores).sum() / positive_count)
