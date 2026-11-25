"""鋼板画像の注釈と提出用 RLE、公式 Dice。"""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SHAPE = (256, 1600)
CLASSES = range(1, 5)


def decode_rle(encoded: str, shape: tuple[int, int] = SHAPE) -> NDArray[np.bool_]:
    pixels = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    values = [int(token) for token in encoded.split()]
    previous_end = 0
    for start, length in zip(values[::2], values[1::2], strict=True):
        offset = start - 1
        end = offset + length
        if offset < previous_end or length < 1 or end > pixels.size:
            raise ValueError(f"不正な RLE 区間 ({start}, {length})")
        pixels[offset:end] = True
        previous_end = end
    return pixels.reshape(shape, order="F")


def encode_rle(mask: NDArray[np.bool_]) -> str:
    pixels = mask.ravel(order="F")
    boundaries = np.flatnonzero(np.diff(pixels, prepend=False, append=False))
    starts = boundaries[::2]
    lengths = boundaries[1::2] - starts
    return " ".join(f"{start + 1} {length}" for start, length in zip(starts, lengths, strict=True))


def dice(target: NDArray[np.bool_], prediction: NDArray[np.bool_]) -> float:
    if target.shape != prediction.shape:
        raise ValueError("マスクの形状が一致しません")
    mass = int(np.count_nonzero(target)) + int(np.count_nonzero(prediction))
    return 2 * int(np.count_nonzero(target & prediction)) / mass if mass else 1.0


def load_labels(data: Path) -> dict[str, dict[int, str]]:
    labels: dict[str, dict[int, str]] = {
        path.name: {} for path in sorted((data / "train_images").glob("*.jpg"))
    }
    if not labels:
        raise ValueError("学習画像がありません")
    with (data / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row["ImageId"]
            class_id = int(row["ClassId"])
            encoded = row["EncodedPixels"]
            if image_id not in labels or class_id not in CLASSES or class_id in labels[image_id]:
                raise ValueError(f"不正または重複した注釈 ({image_id}, {class_id})")
            if not encoded:
                raise ValueError(f"空の注釈 ({image_id}, {class_id})")
            labels[image_id][class_id] = encoded
    return labels
