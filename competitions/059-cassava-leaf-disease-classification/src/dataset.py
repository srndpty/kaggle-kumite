"""公式画像の検証と、近重複をまとめた固定分割。"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
SEED = 42
SPLIT_VERSION = "cassava-group-v1"


def difference_hash(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = np.packbits((pixels[:, 1:] > pixels[:, :-1]).reshape(-1))
    return int.from_bytes(bits.tobytes(), "big"), width, height


def near_duplicate_groups(hashes: list[int]) -> list[int]:
    parents: dict[int, int] = {}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def root(value: int) -> int:
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for value in dict.fromkeys(hashes):
        # 4 ビット以下の差なら、5 区間の少なくとも一つは一致する。
        keys = [
            (offset, (value >> offset) & ((1 << width) - 1))
            for offset, width in ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
        ]
        candidates = {other for key in keys for other in buckets[key]}
        parents[value] = value
        for other in candidates:
            if (value ^ other).bit_count() <= 4:
                left, right = root(value), root(other)
                parents[max(left, right)] = min(left, right)
        for key in keys:
            buckets[key].append(value)
    group_ids: dict[int, int] = {}
    return [group_ids.setdefault(root(value), len(group_ids)) for value in hashes]


def prepare() -> Path:
    rows = pd.read_csv(DATA / "train.csv").sort_values("image_id", ignore_index=True)
    if not rows["image_id"].is_unique or set(rows["label"]) != set(range(5)):
        raise ValueError("公式学習ラベルの画像 ID またはクラスが不正です")
    hashes, widths, heights = zip(
        *(difference_hash(DATA / "train_images" / str(image_id)) for image_id in rows["image_id"]),
        strict=True,
    )
    rows["width"], rows["height"] = widths, heights
    rows["group"] = near_duplicate_groups(list(hashes))
    rows["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(rows, rows["label"], rows["group"])):
        rows.loc[valid, "fold"] = fold
    if (rows["fold"] < 0).any() or (rows.groupby("group")["fold"].nunique() != 1).any():
        raise ValueError("固定分割の検証に失敗しました")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{SPLIT_VERSION}.csv"
    rows.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(prepare())
