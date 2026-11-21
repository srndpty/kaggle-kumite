"""公式データの読み込みと整合性検査。"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from segmentation import SIZE


@dataclass(frozen=True)
class Data:
    train_ids: list[str]
    test_ids: list[str]
    train: np.ndarray
    masks: np.ndarray
    test: np.ndarray
    depths: np.ndarray


def read_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        ids = [row["id"] for row in csv.DictReader(file)]
    if not ids or len(ids) != len(set(ids)) or any(not key for key in ids):
        raise ValueError(f"ID が不正です: {path}")
    return ids


def read_images(directory: Path, ids: list[str], *, masks: bool = False) -> np.ndarray:
    if {path.stem for path in directory.glob("*.png")} != set(ids):
        raise ValueError(f"PNG と CSV の ID が一致しません: {directory}")
    images = np.empty((len(ids), SIZE, SIZE), dtype=np.bool_ if masks else np.uint8)
    for i, key in enumerate(ids):
        with Image.open(directory / f"{key}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
            if pixels.shape != (SIZE, SIZE, 3) or not (
                (pixels[..., 0] == pixels[..., 1]).all()
                and (pixels[..., 0] == pixels[..., 2]).all()
            ):
                raise ValueError(f"画像のサイズまたはチャネルが不正です: {key}")
            gray = pixels[..., 0]
            if masks and not np.isin(gray, [0, 255]).all():
                raise ValueError(f"マスクが二値ではありません: {key}")
            images[i] = gray
    return images


def load_data(root: Path) -> Data:
    train_ids = sorted(read_ids(root / "train.csv"))
    test_ids = read_ids(root / "sample_submission.csv")
    with (root / "depths.csv").open(newline="", encoding="utf-8-sig") as file:
        depths = {row["id"]: int(row["z"]) for row in csv.DictReader(file)}
    if set(train_ids) & set(test_ids) or set(depths) != set(train_ids + test_ids):
        raise ValueError("学習・テスト・深度の ID が不整合です")
    images = root / "competition_data"
    return Data(
        train_ids,
        test_ids,
        read_images(images / "train/images", train_ids),
        read_images(images / "train/masks", train_ids, masks=True),
        read_images(images / "test/images", test_ids),
        np.array([depths[key] for key in train_ids]),
    )
