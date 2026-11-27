"""OTTO の時間窓 CV と保存 OOF の独立再採点。"""

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import polars as pl
from evaluation import (
    ACTIONS,
    Event,
    Window,
    split_session,
    weighted_recall,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "parquet" / "train"
METRIC = "WeightedRecall@20"

Session = tuple[int, list[Event]]


def sessions(window: Window, train: Path = TRAIN, *, every: int = 1) -> Iterator[Session]:
    for partition in sorted(train.glob("*.parquet")):
        frame = (
            pl.scan_parquet(partition)
            .filter(
                (pl.col("start") >= window.start)
                & (pl.col("end") < window.end)
                & (pl.col("events").list.len() >= 2)
                & (pl.col("session") % every == 0)
            )
            .select("session", "events")
            .collect()
        )
        for session, events in frame.iter_rows():
            yield int(session), cast(list[Event], events)


def assess_file(path: Path, windows: tuple[Window, ...]) -> tuple[list[float], float]:
    folds: list[float] = []
    all_counts = {action: [0, 0] for action in ACTIONS}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        rows = csv.reader(source)
        for fold, window in enumerate(windows):
            counts = {action: [0, 0] for action in ACTIONS}
            for session, events in sessions(window):
                _, targets = split_session(session, events)
                for action in ACTIONS:
                    row = next(rows, [])
                    if len(row) != 4 or row[:3] != [str(fold), str(session), action]:
                        raise ValueError(f"OOF のキーが一致しません: {fold}, {session}, {action}")
                    aids = [int(aid) for aid in row[3].split()]
                    if (
                        len(aids) > 20
                        or len(aids) != len(set(aids))
                        or any(aid < 0 for aid in aids)
                    ):
                        raise ValueError(f"無効な OOF 商品リスト: {session}, {action}")
                    counts[action][0] += len(targets[action].intersection(aids))
                    counts[action][1] += min(20, len(targets[action]))
            folds.append(
                weighted_recall({action: (hit, total) for action, (hit, total) in counts.items()})
            )
            for action in ACTIONS:
                all_counts[action][0] += counts[action][0]
                all_counts[action][1] += counts[action][1]
        if next(rows, None) is not None:
            raise ValueError("OOF に余分なキーがあります")
    return folds, weighted_recall(
        {action: (hit, total) for action, (hit, total) in all_counts.items()}
    )
