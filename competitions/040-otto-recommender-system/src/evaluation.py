"""時間窓・疑似切断と OTTO の行動別 Recall@20。"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

ACTIONS = ("clicks", "carts", "orders")
WEIGHTS = (0.10, 0.30, 0.60)
WEEK_MS = 7 * 24 * 60 * 60 * 1000
SEED = 42
SPLIT_VERSION = "weekly-prefix-v1"

Query = tuple[int, str]


class Event(TypedDict):
    aid: int
    ts: int
    type: str


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    def contains(self, start: int, end: int) -> bool:
        return self.start <= start <= end < self.end

    def past_week(self) -> "Window":
        return Window(self.start - WEEK_MS, self.start)


def validation_windows(last_timestamp: int) -> tuple[Window, Window]:
    end = last_timestamp + 1
    last = Window(end - WEEK_MS, end)
    return last.past_week(), last


def split_session(
    session: int, events: Sequence[Event]
) -> tuple[Sequence[Event], dict[str, set[int]]]:
    if len(events) < 2:
        raise ValueError("疑似切断には2イベント以上必要です")
    digest = hashlib.blake2b(f"{SEED}:{session}".encode(), digest_size=8).digest()
    cut = 1 + int.from_bytes(digest, "little") % (len(events) - 1)
    truth: dict[str, set[int]] = {action: set() for action in ACTIONS}
    for event in events[cut:]:
        action = event["type"]
        if action != "clicks" or not truth[action]:
            truth[action].add(event["aid"])
    return events[:cut], truth


def recall_counts(
    truth: Mapping[Query, set[int]], predictions: Iterable[tuple[Query, Sequence[int]]]
) -> dict[str, tuple[int, int]]:
    counts = {action: (0, 0) for action in ACTIONS}
    seen: set[Query] = set()
    for key, aids in predictions:
        if key in seen or key not in truth:
            raise ValueError(f"重複または未知の評価キー: {key}")
        seen.add(key)
        if len(aids) > 20 or len(aids) != len(set(aids)) or any(aid < 0 for aid in aids):
            raise ValueError(f"無効な商品リスト: {key}")
        action = key[1]
        hits, denominator = counts[action]
        target = truth[key]
        counts[action] = hits + len(target.intersection(aids)), denominator + min(20, len(target))
    if len(seen) != len(truth):
        raise ValueError("評価キーが欠落しています")
    return counts


def weighted_recall(counts: Mapping[str, tuple[int, int]]) -> float:
    score = 0.0
    for action, weight in zip(ACTIONS, WEIGHTS, strict=True):
        hits, denominator = counts[action]
        if denominator <= 0 or not 0 <= hits <= denominator:
            raise ValueError(f"不正な評価件数: {action}")
        score += weight * hits / denominator
    return score
