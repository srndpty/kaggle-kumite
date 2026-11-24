"""画像単位の Global Wheat Detection 評価。"""

import numpy as np
from numpy.typing import NDArray

THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def image_score(
    truth: NDArray[np.float64],
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
) -> float | None:
    """xyxy の矩形を信頼度順に照合する。両方が空の画像は集計対象外。"""
    if len(truth) == 0 or len(boxes) == 0:
        return 0.0 if len(truth) + len(boxes) else None
    return float(
        np.mean(
            [
                count / (len(truth) + len(boxes) - count)
                for count in match_counts(truth, boxes, scores, THRESHOLDS)
            ]
        )
    )


def match_counts(
    truth: NDArray[np.float64],
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    thresholds: tuple[float, ...],
) -> list[int]:
    if not len(truth) or not len(boxes):
        return [0] * len(thresholds)
    ordered = boxes[np.argsort(-scores, kind="stable")]
    lower = np.maximum(ordered[:, None, :2], truth[None, :, :2])
    upper = np.minimum(ordered[:, None, 2:], truth[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    predicted_area = (ordered[:, 2:] - ordered[:, :2]).prod(axis=1)
    true_area = (truth[:, 2:] - truth[:, :2]).prod(axis=1)
    iou = intersection / (predicted_area[:, None] + true_area[None, :] - intersection)

    counts: list[int] = []
    for threshold in thresholds:
        available = np.ones(len(truth), dtype=bool)
        matches = 0
        for overlaps in iou:
            candidates = np.where(available, overlaps, -1)
            best = int(candidates.argmax())
            if candidates[best] > threshold:
                available[best] = False
                matches += 1
        counts.append(matches)
    return counts
