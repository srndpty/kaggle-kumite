"""比較可能な run の選定と、撮影元・穂の密度・サイズ別の誤差集計。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dataset import ROOT, load_samples
from metric import match_counts

from kumite.provenance import files_sha256
from kumite.runs import RunStore


def analyze(baseline: str, candidates: list[str], output: Path) -> None:
    store = RunStore(ROOT)
    records = [store.record(run_id) for run_id in [baseline, *candidates]]
    base = records[0]
    if base.params.get("trial") != "E00" or base.split_version != "source-v1":
        raise ValueError("Baseline must be E00 with source-v1")
    if any(
        record.data_hash != base.data_hash
        or record.split_hash != base.split_hash
        or record.split_version != base.split_version
        or record.metric != base.metric
        for record in records
    ):
        raise ValueError("Selection requires comparable source-v1 records")
    selected = min(
        records, key=lambda record: (-record.oof_score, record.run_id != baseline, record.run_id)
    )
    data = ROOT / "data"
    samples, _ = load_samples(data)
    if files_sha256([data / "train.csv", *(sample.path for sample in samples)]) != base.data_hash:
        raise ValueError("Training data changed since CV")
    labels = {sample.path.stem: sample.boxes for sample in samples}

    def read(run_id: str) -> dict[str, dict[str, object]]:
        path = store.artifact_path(run_id, "oof.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {row["image_id"]: row for row in rows}
        if len(rows) != len(by_id) or by_id.keys() != labels.keys():
            raise ValueError(f"Unexpected OOF image IDs: {run_id}")
        return by_id

    reference = read(baseline)
    comparisons: list[dict[str, object]] = []
    for record in records:
        groups: dict[str, list[float]] = defaultdict(list)
        differences: dict[str, list[float]] = defaultdict(list)
        errors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for key, row in read(record.run_id).items():
            truth = labels[key]
            count = len(truth)
            density = "0-20" if count <= 20 else "21-50" if count <= 50 else "51+"
            median_area = (
                float(np.median([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in truth]))
                if truth
                else 0.0
            )
            size = "none" if not truth else "<5500" if median_area < 5500 else ">=5500"
            boxes = np.asarray(row["boxes"], dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(row["scores"], dtype=np.float64)
            matches = match_counts(np.asarray(truth).reshape(-1, 4), boxes, scores, (0.5,))[0]
            value = row["image_score"]
            ref = reference[key]["image_score"]
            for group in (
                "all",
                f"source/{row['source']}",
                f"density/{density}",
                f"median-box-area/{size}",
            ):
                errors[group][0] += len(boxes) - matches
                errors[group][1] += count - matches
                if isinstance(value, (int, float)):
                    groups[group].append(float(value))
                    if isinstance(ref, (int, float)):
                        differences[group].append(float(value) - float(ref))
        comparisons.append(
            {
                "run_id": record.run_id,
                "oof_score": record.oof_score,
                "groups": {
                    group: {
                        "images": len(values),
                        "mean_score": float(np.mean(values)),
                        "false_positive_at_iou_50": errors[group][0],
                        "false_negative_at_iou_50": errors[group][1],
                        "paired_delta": float(np.mean(differences[group]))
                        if differences[group]
                        else None,
                    }
                    for group, values in sorted(groups.items())
                },
            }
        )

    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "selected_run": selected.run_id,
                "data_hash": base.data_hash,
                "split_hash": base.split_hash,
                "comparisons": comparisons,
            },
            stream,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    analyze(args.baseline, args.candidates, args.output)


if __name__ == "__main__":
    main()
