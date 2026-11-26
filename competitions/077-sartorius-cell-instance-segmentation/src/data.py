"""公式画像の読み込みと撮影サンプル単位の分割。"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = ROOT.parents[1]
SEED = 42


@dataclass
class Image:
    id: str
    path: Path
    width: int
    height: int
    cell_type: str
    sample: str
    masks: list[str]

    @property
    def size(self) -> int:
        return self.width * self.height


def dataset(data: Path = DATA) -> tuple[list[Image], list[int], str, str]:
    annotations = (data / "train.csv").read_bytes()
    images: dict[str, Image] = {}
    for row in csv.DictReader(annotations.decode("utf-8-sig").splitlines()):
        image_id = row["id"]
        image = images.setdefault(
            image_id,
            Image(
                image_id,
                data / "train" / f"{image_id}.png",
                int(row["width"]),
                int(row["height"]),
                row["cell_type"],
                row["sample_id"],
                [],
            ),
        )
        if (image.width, image.height, image.cell_type, image.sample) != (
            int(row["width"]),
            int(row["height"]),
            row["cell_type"],
            row["sample_id"],
        ):
            raise ValueError(f"画像内のメタデータが一致しません: {image_id}")
        image.masks.append(row["annotation"])
    if set(images) != {path.stem for path in (data / "train").glob("*.png")}:
        raise ValueError("画像と注釈の ID が一致しません")
    rows = [images[image_id] for image_id in sorted(images)]
    parents = {image.sample: image.sample for image in rows}

    def root(sample: str) -> str:
        while parents[sample] != sample:
            sample = parents[sample]
        return sample

    digest = hashlib.sha256(annotations)
    first_sample: dict[bytes, str] = {}
    for image in rows:
        image_hash = hashlib.sha256(image.path.read_bytes()).digest()
        digest.update(image_hash)
        previous = first_sample.setdefault(image_hash, image.sample)
        parents[root(image.sample)] = root(previous)
    groups = [root(image.sample) for image in rows]
    classes = [image.cell_type for image in rows]
    folds = [-1] * len(rows)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (train, valid) in enumerate(splitter.split(rows, classes, groups)):
        if {groups[i] for i in train} & {groups[i] for i in valid}:
            raise ValueError("サンプルが学習と検証にまたがっています")
        for index in valid:
            folds[index] = fold
    if set(folds) != {0, 1, 2}:
        raise ValueError("検証分割が不完全です")
    split = json.dumps(list(zip(sorted(images), folds, strict=True))).encode()
    return rows, folds, digest.hexdigest(), hashlib.sha256(split).hexdigest()
