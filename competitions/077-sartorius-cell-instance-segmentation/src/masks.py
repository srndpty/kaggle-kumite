"""行優先 RLE、重複除去、画像単位の IoU 採点。"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def pixels(rle: str, size: int) -> NDArray[np.int64]:
    values = [int(value) for value in rle.split()]
    if not values or len(values) % 2:
        raise ValueError("RLE に開始位置と長さの組が必要です")
    indices: list[NDArray[np.int64]] = []
    end = 0
    for start, length in zip(values[::2], values[1::2], strict=True):
        offset = start - 1
        if offset < end or length <= 0 or offset + length > size:
            raise ValueError("RLE の順序または範囲が不正です")
        end = offset + length
        indices.append(np.arange(offset, end, dtype=np.int64))
    return np.concatenate(indices)


def encode(indices: NDArray[np.int64]) -> str:
    if not len(indices):
        return ""
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return " ".join(f"{run[0] + 1} {len(run)}" for run in runs)


def disjoint(candidates: list[tuple[float, str]], size: int, retain: float) -> list[str]:
    occupied = np.zeros(size, dtype=np.bool_)
    result: list[str] = []
    for confidence, rle in sorted(candidates, key=lambda candidate: -candidate[0]):
        if not 0 <= confidence <= 1:
            raise ValueError("予測スコアが範囲外です")
        original = pixels(rle, size)
        remaining = original[~occupied[original]]
        if len(remaining) and len(remaining) >= len(original) * retain:
            occupied[remaining] = True
            result.append(encode(remaining))
    return result


def ious(truth: list[str], prediction: list[str], size: int) -> NDArray[np.float64]:
    labels = np.zeros(size, dtype=np.int32)
    areas: list[int] = []
    for label, rle in enumerate(prediction, start=1):
        indices = pixels(rle, size)
        if np.any(labels[indices]):
            raise ValueError("同一画像の予測マスクが重複しています")
        labels[indices] = label
        areas.append(len(indices))
    result = np.zeros((len(truth), len(prediction)), dtype=np.float64)
    for row, rle in enumerate(truth):
        indices = pixels(rle, size)
        intersection = np.bincount(labels[indices], minlength=len(areas) + 1)[1:]
        result[row] = intersection / (len(indices) + np.asarray(areas) - intersection)
    return result


def image_score(overlap: NDArray[np.float64]) -> float:
    count = sum(overlap.shape)
    if not count:
        return 1.0
    scores: list[float] = []
    for percent in range(50, 100, 5):
        matches = maximum_bipartite_matching(csr_matrix(overlap > percent / 100))
        hits = int(np.count_nonzero(matches >= 0))
        scores.append(hits / (count - hits))
    return float(np.mean(scores))
