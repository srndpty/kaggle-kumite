"""公式音源の検査・重複グループ化・検証分割。"""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile

from kumite.group_folds import multilabel_group_folds
from kumite.provenance import array_sha256, files_sha256

RATE = 16000
WINDOW = 2 * RATE
FOLDS = 3
SEED = 42


@dataclass(frozen=True)
class Clip:
    path: Path
    group: str
    labels: tuple[str, ...]
    seconds: float
    gain_group: str | None = None


def data_fingerprint(root: Path, clips: list[Clip]) -> str:
    metadata = files_sha256(
        [root / name for name in ("sample_submission.csv", "train_curated.csv", "train_noisy.csv")]
    )
    fingerprints = np.array([clip.group for clip in clips])
    return array_sha256(np.array([metadata, array_sha256(fingerprints)]))


def read_clips(root: Path, source: str, *, check_gain: bool = False) -> list[Clip]:
    rows = pd.read_csv(root / f"train_{source}.csv", dtype=str)
    if rows["fname"].isna().any() or rows["fname"].duplicated().any():
        raise ValueError(f"Invalid clip IDs in {source}")
    clips = []
    for row in rows.itertuples(index=False):
        if not isinstance(row.fname, str) or not isinstance(row.labels, str) or not row.labels:
            raise ValueError(f"Invalid metadata row in {source}")
        path = root / row.fname
        wave, rate = soundfile.read(path, dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f"Invalid audio: {path}")
        digest = hashlib.sha256(f"{rate}/{wave.shape}:".encode())
        digest.update(wave.tobytes())
        gain_group = None
        if check_gain:
            centered = wave - wave[0]
            peak = centered.flat[int(np.abs(centered).argmax())]
            if peak != 0:
                quantized = np.rint(centered / peak * 4096).astype("<i2")
                normalized = hashlib.sha256(f"{rate}/{wave.shape}:".encode())
                normalized.update(quantized.tobytes())
                gain_group = normalized.hexdigest()
        clips.append(
            Clip(
                path, digest.hexdigest(), tuple(row.labels.split(",")), len(wave) / rate, gain_group
            )
        )
    return clips


def targets(clips: list[Clip], tags: list[str]) -> npt.NDArray[np.bool_]:
    columns = {tag: i for i, tag in enumerate(tags)}
    labels = np.zeros((len(clips), len(tags)), dtype=bool)
    for i, clip in enumerate(clips):
        labels[i, [columns[tag] for tag in clip.labels]] = True
    return labels


def split_groups(clips: list[Clip], labels: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    return multilabel_group_folds([clip.group for clip in clips], labels, n_folds=FOLDS, seed=SEED)


def eligible_noisy(noisy: list[Clip], validation: list[Clip]) -> list[Clip]:
    excluded = {clip.group for clip in validation}
    return [clip for clip in noisy if clip.group not in excluded]


def read_wave(path: Path | BytesIO) -> npt.NDArray[np.float32]:
    wave, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = wave.mean(axis=1)
    if rate != RATE:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=RATE)
    return np.asarray(mono, dtype=np.float32)


def windows(wave: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if len(wave) <= WINDOW:
        return np.pad(wave, (0, WINDOW - len(wave)))[None, :]
    last = len(wave) - WINDOW
    starts = range(0, last, RATE)
    return np.stack([wave[i : i + WINDOW] for i in (*starts, last)])
