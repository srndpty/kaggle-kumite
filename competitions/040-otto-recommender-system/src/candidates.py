"""候補順と教師の採用条件。"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation import ACTIONS, Event


@dataclass(frozen=True)
class Method:
    width: int
    training_hit_width: int
    elapsed_feature: bool = False


METHODS = {
    "B00": Method(20, 0),
    "E01": Method(40, 40),
    "E02": Method(100, 40),
    "E03": Method(100, 100),
    "E04": Method(100, 40, elapsed_feature=True),
}


def candidate_aids(prefix: Sequence[Event], popular: Sequence[int], width: int) -> list[int]:
    candidates = dict.fromkeys(event["aid"] for event in reversed(prefix))
    for aid in popular:
        if len(candidates) >= width:
            break
        candidates[aid] = None
    return list(candidates)[:width]


def rank_features(
    prefix: Sequence[Event],
    candidates: Sequence[int],
    popularity: Mapping[int, int],
    *,
    elapsed_feature: bool = False,
) -> list[list[float]]:
    lag: dict[int, int] = {}
    last_time: dict[int, int] = {}
    frequencies = {action: Counter[int]() for action in ACTIONS}
    for position, event in enumerate(reversed(prefix)):
        aid = event["aid"]
        lag.setdefault(aid, position)
        if elapsed_feature:
            last_time.setdefault(aid, event["ts"])
        frequencies[event["type"]][aid] += 1
    rows = [
        [
            float(rank),
            float(lag.get(aid, len(prefix))),
            float(popularity.get(aid, 0)),
            *(float(frequencies[action][aid]) for action in ACTIONS),
        ]
        for rank, aid in enumerate(candidates)
    ]
    if elapsed_feature:
        now = prefix[-1]["ts"]
        for aid, features in zip(candidates, rows, strict=True):
            features.append((now - last_time[aid]) / 1000 if aid in last_time else -1.0)
    return rows


def training_labels(candidates: Sequence[int], truth: set[int], method: Method) -> list[int] | None:
    labels = [int(aid in truth) for aid in candidates]
    return labels if any(labels[: method.training_hit_width]) else None
