"""元のノートの文字位置で区間を検証し、公式 micro-F1 を計算する。"""

from collections.abc import Iterable

type Span = tuple[int, int]


def normalize(spans: Iterable[Span], length: int) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if not 0 <= start < end <= length:
            msg = f"文字区間が本文の範囲外です: {(start, end)} / {length}"
            raise ValueError(msg)
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def characters(spans: Iterable[Span], length: int) -> set[int]:
    return {index for start, end in normalize(spans, length) for index in range(start, end)}


def micro_f1(rows: Iterable[tuple[int, list[Span], list[Span]]]) -> float:
    matched = expected = predicted = 0
    for length, truth, prediction in rows:
        truth_chars = characters(truth, length)
        prediction_chars = characters(prediction, length)
        matched += len(truth_chars & prediction_chars)
        expected += len(truth_chars)
        predicted += len(prediction_chars)
    return 2 * matched / (expected + predicted) if expected + predicted else 0.0
