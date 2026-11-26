"""公式波形の検査と CV 分割。"""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from kumite.crossfit import Split

SEED = 42
SHAPE = (3, 4096)
SAMPLE_RATE = 2048


def read_ids(path: Path, *, training: bool = False) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "target"]:
            raise ValueError(f"列が不正です: {path}")
        ids = []
        labels = []
        for row in reader:
            if None in row or row["id"] is None or row["target"] is None:
                raise ValueError(f"行の列数が不正です: {path}")
            ids.append(row["id"])
            if training:
                label = float(row["target"])
                if label not in (0, 1):
                    raise ValueError(f"ラベルが不正です: {path}")
                labels.append(int(label))
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not re.fullmatch(r"[0-9a-f]{10}", key) for key in ids)
    ):
        raise ValueError(f"ID が不正です: {path}")
    return ids, np.array(labels, dtype=np.int64)


def waveform_path(data: Path, split: str, key: str) -> Path:
    return data.joinpath(split, *key[:3], f"{key}.npy")


def validate_waveform(wave: np.ndarray, name: str) -> None:
    if wave.shape != SHAPE or wave.dtype.kind != "f" or not np.isfinite(wave).all():
        raise ValueError(f"波形が不正です: {name}")


def fingerprint(wave: np.ndarray) -> str:
    canonical = np.asarray(wave, dtype="<f8").copy()
    canonical[canonical == 0] = 0
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def make_splits(y: np.ndarray, groups: np.ndarray) -> list[Split]:
    if not np.isin(y, (0, 1)).all() or len(y) != len(groups):
        raise ValueError("ラベルまたはグループが不正です")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, valid in splits:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("学習側と検証側に両クラスが必要です")
        if np.intersect1d(groups[train], groups[valid]).size:
            raise ValueError("グループが学習側と検証側をまたいでいます")
    if not np.array_equal(
        np.sort(np.concatenate([valid for _, valid in splits])), np.arange(len(y))
    ):
        raise ValueError("検証側の割り当てが不正です")
    return splits
