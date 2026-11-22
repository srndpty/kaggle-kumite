"""公式データの読み込みと画像集約。"""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import DisjointSet

ID = "PetID"
GROUP = "RescuerID"
TARGET = "AdoptionSpeed"
CATEGORIES = (
    "Type",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "Health",
    "State",
)
IMAGE_STATS = ("width", "height", "aspect_ratio", "brightness")


def read_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    paths = [
        root / "train" / "train.csv",
        root / "test" / "test.csv",
        root / "test" / "sample_submission.csv",
    ]
    train, test, sample = (pd.read_csv(path, dtype={ID: str, GROUP: str}) for path in paths)
    for frame in (train, test, sample):
        if frame[ID].isna().any() or not frame[ID].is_unique:
            raise ValueError("PetID に欠損または重複があります")
    if train[GROUP].isna().any() or not train[TARGET].isin(range(5)).all():
        raise ValueError("学習データの保護者 ID またはラベルが無効です")
    if set(test[ID]) != set(sample[ID]) or set(train[ID]) & set(test[ID]):
        raise ValueError("train、test、sample_submission の PetID が不整合です")
    return train, test, sample, paths


def image_paths(root: Path, split: str) -> list[Path]:
    folder = root / f"{split}_images"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(folder.glob("*.jpg"))


def image_features(
    paths: list[Path], ids: pd.Series, *, summarize: bool
) -> tuple[pd.DataFrame, dict[str, str], int]:
    rows = []
    hashes = {}
    unreadable = 0
    known = set(ids)
    for path in paths:
        pet, _ = path.stem.rsplit("-", 1)
        if pet not in known:
            raise ValueError(f"画像に対応する PetID がありません: {path.name}")
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        if summarize:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    gray = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
                    rows.append(
                        (pet, width, height, width / height, float(np.asarray(gray).mean()))
                    )
            except (OSError, ValueError):
                unreadable += 1
    grouped = pd.DataFrame(rows, columns=[ID, *IMAGE_STATS]).groupby(ID)
    features = grouped[list(IMAGE_STATS)].mean().reindex(ids)
    features["readable_images"] = grouped.size().reindex(ids, fill_value=0)
    return features.reset_index(drop=True), hashes, unreadable


def connected_groups(train: pd.DataFrame, hashes: dict[str, str]) -> np.ndarray:
    rescuers = dict(zip(train[ID], train[GROUP], strict=True))
    groups = DisjointSet(sorted(set(rescuers.values())))
    first_owner: dict[str, str] = {}
    for name, digest in hashes.items():
        owner = rescuers[Path(name).stem.rsplit("-", 1)[0]]
        groups.merge(first_owner.setdefault(digest, owner), owner)
    return np.array([groups[owner] for owner in train[GROUP]])


def tabular(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(
        columns=[ID, GROUP, TARGET, "Name", "Description"], errors="ignore"
    ).copy()
    for column in CATEGORIES:
        features[column] = features[column].astype("string").fillna("<欠損>").astype(str)
    return features
