"""TGS のマスク採点、提出、検証分割。"""

import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SIZE = 101
SEED = 42


def image_scores(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if target.shape != predicted.shape or target.ndim != 3:
        raise ValueError("マスクの形状が一致しません")
    intersection = np.count_nonzero(target & predicted, axis=(1, 2))
    union = np.count_nonzero(target | predicted, axis=(1, 2))
    hits = intersection[:, None] * 20 > union[:, None] * np.arange(10, 20)
    return np.where(union == 0, 1.0, hits.mean(axis=1))


def score(target: np.ndarray, predicted: np.ndarray) -> float:
    return float(image_scores(target, predicted.astype(bool)).mean())


def encode(mask: np.ndarray) -> str:
    pixels = mask.ravel(order="F").astype(np.int8)
    edges = np.flatnonzero(np.diff(pixels, prepend=0, append=0))
    runs = zip(edges[::2] + 1, edges[1::2] - edges[::2], strict=True)
    return " ".join(f"{start} {length}" for start, length in runs)


def write_submission(path: Path, ids: list[str], masks: np.ndarray) -> None:
    if masks.shape != (len(ids), SIZE, SIZE) or masks.dtype != np.bool_:
        raise ValueError("提出マスクの形状または型が不正です")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "rle_mask"))
        writer.writerows((key, encode(mask)) for key, mask in zip(ids, masks, strict=True))


def duplicate_groups(images: np.ndarray) -> np.ndarray:
    return np.array(
        [
            hashlib.sha256(min(image.tobytes(), image[:, ::-1].tobytes())).hexdigest()
            for image in images
        ]
    )


def coverage_bins(masks: np.ndarray) -> np.ndarray:
    return np.searchsorted([0, 0.1, 0.5], masks.mean(axis=(1, 2)))


def make_folds(masks: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED)
    folds = np.full(len(masks), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(masks, coverage_bins(masks), groups)):
        folds[valid] = fold
    return folds


def choose_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted(range(30, 71, 5), key=lambda value: (abs(value - 50), value))
    return max(
        (value / 100 for value in candidates),
        key=lambda threshold: score(target, probabilities >= threshold),
    )
