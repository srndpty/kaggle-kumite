"""録音と区間注釈から学習・推論窓を作る。"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 32000
WINDOW = 5 * RATE
SPECIES = tuple(f"s{i}" for i in range(24))
ID = "recording_id"


@dataclass(frozen=True)
class Event:
    species: int
    start: int
    end: int


@dataclass(frozen=True)
class Example:
    row: int
    start: int
    species: tuple[int, ...]


def read_events(path: Path) -> dict[str, list[Event]]:
    events: dict[str, list[Event]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row[ID]
            species = int(row["species_id"])
            start = round(float(row["t_min"]) * RATE)
            end = round(float(row["t_max"]) * RATE)
            if not key or not 0 <= species < len(SPECIES) or not 0 <= start < end:
                raise ValueError(f"不正な注釈: {row}")
            events.setdefault(key, []).append(Event(species, start, end))
    return events


def read_wave(path: Path) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError(f"不正な音声: {path}")
    return resample_poly(mono, RATE, rate).astype(np.float32, copy=False)


def starts(length: int, hop: int) -> list[int]:
    last = max(0, length - WINDOW)
    grid = list(range(0, last + 1, hop))
    if grid[-1] != last:
        grid.append(last)
    return grid


def crop(wave: np.ndarray, start: int) -> np.ndarray:
    window = wave[start : start + WINDOW]
    return np.pad(window, (0, WINDOW - len(window))) if len(window) < WINDOW else window


def overlaps(event: Event, start: int) -> bool:
    return start < event.end and event.start < start + WINDOW


def centered(event: Event, length: int) -> int:
    return min(max(0, length - WINDOW), max(0, (event.start + event.end - WINDOW) // 2))


def training_examples(
    keys: list[str],
    lengths: np.ndarray,
    rows: np.ndarray,
    true: dict[str, list[Event]],
    false: dict[str, list[Event]],
    seed: int,
    replace_background: bool,
) -> tuple[list[Example], list[Example], list[Example]]:
    rng = np.random.default_rng(seed)
    positives: list[Example] = []
    background: list[Example] = []
    hard: list[Example] = []
    for row in rows:
        key = keys[row]
        length = int(lengths[row])
        tp = true.get(key, [])
        fp = false.get(key, [])
        for start in sorted({centered(event, length) for event in tp}):
            species = tuple(sorted({e.species for e in tp if overlaps(e, start)}))
            positives.append(Example(int(row), start, species))
        clear = [
            start
            for start in starts(length, WINDOW)
            if not any(overlaps(event, start) for event in (*tp, *fp))
        ]
        if clear:
            background.append(Example(int(row), int(rng.choice(clear)), ()))
        for event in fp:
            start = centered(event, length)
            if not any(e.species == event.species and overlaps(e, start) for e in tp):
                hard.append(Example(int(row), start, (event.species,)))
    if not positives or not background or (replace_background and not hard):
        raise ValueError("学習窓の必須プールが空です")
    if replace_background:
        n_replace = len(background) // 2
        chosen = set(rng.choice(len(background), n_replace, replace=False).tolist())
        background = [item for i, item in enumerate(background) if i not in chosen]
        hard = [hard[i] for i in rng.choice(len(hard), n_replace, replace=n_replace > len(hard))]
    else:
        hard = []
    return positives, background, hard


def aggregate(scores: np.ndarray, top: int) -> np.ndarray:
    n = min(len(scores), top)
    return np.sort(scores, axis=0)[-n:].mean(axis=0)
