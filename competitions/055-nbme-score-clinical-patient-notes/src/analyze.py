"""選定済み run の保存 OOF からケース別・誤差種別の集計を出す。"""

import argparse
import json
from collections import Counter, defaultdict

from data import ROOT
from spans import Span, characters, micro_f1
from train import PREDICTIONS, assess, dataset

from kumite.runs import RunStore


def analyze(run_id: str) -> dict[str, object]:
    store = RunStore(ROOT)
    record = store.record(run_id)
    _, score = assess(lambda name: store.artifact_path(run_id, name), record)
    pairs, _, _, _ = dataset()
    rows = json.loads(store.artifact_path(run_id, PREDICTIONS).read_text(encoding="utf-8"))
    by_case: dict[str, list[tuple[int, list[Span], list[Span]]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for pair, row in zip(pairs, rows, strict=True):
        prediction = [(start, end) for start, end in row["spans"]]
        by_case[pair.case].append((len(pair.text), pair.spans, prediction))
        if pair.spans and not prediction:
            counts["missed_nonempty"] += 1
        if prediction and not pair.spans:
            counts["predicted_on_empty"] += 1
        if pair.spans and prediction:
            for start, end in pair.spans:
                overlapping = [(a, b) for a, b in prediction if a < end and start < b]
                if len(overlapping) > 1:
                    counts["fragmented_reference_span"] += 1
                elif overlapping and overlapping[0] != (start, end):
                    counts["boundary_mismatch_reference_span"] += 1
                elif not overlapping:
                    counts["uncovered_reference_span"] += 1
        expected = characters(pair.spans, len(pair.text))
        actual = characters(prediction, len(pair.text))
        counts["false_positive_characters"] += len(actual - expected)
        counts["false_negative_characters"] += len(expected - actual)
    return {
        "run_id": run_id,
        "oof_micro_f1": score,
        "by_case": {case: micro_f1(values) for case, values in sorted(by_case.items())},
        "counts": dict(counts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_id), indent=2))
