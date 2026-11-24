"""ラベルと撮影元を画像単位にまとめ、比較用の分割を作る。"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold, KFold

from kumite.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sample:
    path: Path
    source: str
    boxes: list[list[float]]


def load_samples(
    data: Path, *, include_unlabeled: bool = False
) -> tuple[list[Sample], dict[str, int]]:
    labels: dict[str, list[list[float]]] = defaultdict(list)
    metadata: dict[str, tuple[str, int, int]] = {}
    with (data / "train.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["image_id"]
            source, width, height = row["source"], int(row["width"]), int(row["height"])
            if metadata.setdefault(key, (source, width, height)) != (source, width, height):
                raise ValueError(f"Inconsistent image metadata: {key}")
            x, y, w, h = map(float, json.loads(row["bbox"]))
            if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                raise ValueError(f"Invalid box: {key}")
            left, top = max(x, 0.0), max(y, 0.0)
            right, bottom = min(x + w, float(width)), min(y + h, float(height))
            if left < right and top < bottom:
                labels[key].append([left, top, right, bottom])

    images = {path.stem: path for path in sorted((data / "train").glob("*.jpg"))}
    if missing := metadata.keys() - images.keys():
        raise ValueError(f"Missing labeled JPEGs: {sorted(missing)[:5]}")
    duplicates: dict[str, list[str]] = defaultdict(list)
    for key, path in images.items():
        duplicates[file_sha256(path)].append(key)
    removed = {key for keys in duplicates.values() if len(keys) > 1 for key in keys}
    samples: list[Sample] = []
    unknown = 0
    for key, path in images.items():
        source, width, height = metadata.get(key, ("", 0, 0))
        if not source:
            unknown += 1
            if include_unlabeled and key not in metadata and key not in removed:
                samples.append(Sample(path, "", []))
        elif key not in removed:
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f"Unexpected JPEG dimensions: {path}")
            samples.append(Sample(path, source, labels[key]))
    return samples, {"unknown_source": unknown, "exact_duplicate": len(removed)}


def split_images(samples: list[Sample], version: str) -> list[int]:
    rows = np.arange(len(samples))
    if version == "source-v1":
        folds = GroupKFold(n_splits=3).split(rows, groups=[sample.source for sample in samples])
    elif version == "random-v1":
        folds = KFold(n_splits=3, shuffle=True, random_state=42).split(rows)
    else:
        raise ValueError(f"Unknown split version: {version}")
    assignment = [-1] * len(rows)
    for fold, (_, valid) in enumerate(folds):
        for index in valid:
            assignment[int(index)] = fold
    return assignment
