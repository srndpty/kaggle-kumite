"""比較可能な run の選定と、撮影元・穂の密度・サイズ別の誤差集計。"""

import argparse
import json
from collections import defaultdict
from math import isclose
from pathlib import Path

import numpy as np
from dataset import ROOT, load_samples, split_images
from metric import image_score, match_counts

from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


def check_score(saved: object, actual: float | None, context: str) -> None:
    if actual is None:
        valid = saved is None
    else:
        valid = (
            type(saved) in (int, float)
            and isinstance(saved, (int, float))
            and isclose(saved, actual, rel_tol=0, abs_tol=1e-12)
        )
    if not valid:
        raise ValueError(f"Score mismatch: {context}")


def build_selection(baseline: str, candidates: list[str]) -> dict[str, object]:
    if len(set([baseline, *candidates])) != 1 + len(candidates):
        raise ValueError("Duplicate selection run IDs")
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
        or record.seed != base.seed
        or not record.greater_is_better
        for record in records
    ):
        raise ValueError("Selection requires comparable source-v1 records")
    data = ROOT / "data"
    samples, _ = load_samples(data)
    if files_sha256([data / "train.csv", *(sample.path for sample in samples)]) != base.data_hash:
        raise ValueError("Training data changed since CV")
    labels = {sample.path.stem: sample for sample in samples}
    folds = split_images(samples, base.split_version)
    expected_split = dict(zip(labels, folds, strict=True))
    if array_sha256(np.asarray(folds, dtype=np.int64)) != base.split_hash:
        raise ValueError("Fixed split hash mismatch")

    def read(run_id: str) -> dict[str, dict[str, object]]:
        path = store.artifact_path(run_id, "oof.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {row["image_id"]: row for row in rows}
        if len(rows) != len(by_id) or by_id.keys() != labels.keys():
            raise ValueError(f"Unexpected OOF image IDs: {run_id}")
        return by_id

    reference: dict[str, float | None] = {}
    rescored: dict[str, float] = {}
    comparisons: list[dict[str, object]] = []
    for record in records:
        split = json.loads(
            store.artifact_path(record.run_id, "split.json").read_text(encoding="utf-8")
        )
        if split != expected_split:
            raise ValueError(f"Invalid fixed split: {record.run_id}")
        groups: dict[str, list[float]] = defaultdict(list)
        differences: dict[str, list[float]] = defaultdict(list)
        errors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        fold_scores: dict[int, list[float]] = defaultdict(list)
        for key, row in read(record.run_id).items():
            sample = labels[key]
            truth = sample.boxes
            fold = expected_split[key]
            if (
                type(row["fold"]) is not int
                or row["fold"] != fold
                or row["source"] != sample.source
            ):
                raise ValueError(f"Invalid OOF fold or source: {record.run_id}/{key}")
            count = len(truth)
            density = "0-20" if count <= 20 else "21-50" if count <= 50 else "51+"
            median_area = (
                float(np.median([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in truth]))
                if truth
                else 0.0
            )
            size = "none" if not truth else "<5500" if median_area < 5500 else ">=5500"
            raw_boxes, raw_scores = row["boxes"], row["scores"]
            if (
                not isinstance(raw_boxes, list)
                or not isinstance(raw_scores, list)
                or any(
                    not isinstance(box, list)
                    or len(box) != 4
                    or any(type(x) not in (int, float) for x in box)
                    for box in raw_boxes
                )
                or any(type(score) not in (int, float) for score in raw_scores)
            ):
                raise ValueError(f"Invalid OOF prediction shape: {record.run_id}/{key}")
            boxes = np.asarray(raw_boxes, dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(raw_scores, dtype=np.float64)
            if (
                len(boxes) != len(scores)
                or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all()
                or (scores < 0).any()
                or (scores > 1).any()
                or (boxes < 0).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError(f"Invalid OOF detections: {record.run_id}/{key}")
            truth_array = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
            value = image_score(truth_array, boxes, scores)
            check_score(row["image_score"], value, f"{record.run_id}/{key}")
            if record.run_id == baseline:
                reference[key] = value
            ref = reference[key]
            if value is not None:
                fold_scores[fold].append(value)
            matches = match_counts(truth_array, boxes, scores, (0.5,))[0]
            for group in (
                "all",
                f"source/{row['source']}",
                f"density/{density}",
                f"median-box-area/{size}",
            ):
                errors[group][0] += len(boxes) - matches
                errors[group][1] += count - matches
                if value is not None:
                    groups[group].append(value)
                    if ref is not None:
                        differences[group].append(value - ref)
        oof = float(np.mean(groups["all"]))
        check_score(record.oof_score, oof, record.run_id)
        if len(record.fold_scores) != len(set(folds)):
            raise ValueError(f"Invalid number of fold scores: {record.run_id}")
        for fold, saved in enumerate(record.fold_scores):
            check_score(saved, float(np.mean(fold_scores[fold])), f"{record.run_id}/fold {fold}")
        rescored[record.run_id] = oof
        comparisons.append(
            {
                "run_id": record.run_id,
                "oof_score": oof,
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

    selected = min(rescored, key=lambda run_id: (-rescored[run_id], run_id != baseline, run_id))
    return {
        "selected_run": selected,
        "data_hash": base.data_hash,
        "split_hash": base.split_hash,
        "comparisons": comparisons,
    }


def analyze(baseline: str, candidates: list[str], output: Path) -> None:
    selection = build_selection(baseline, candidates)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    analyze(args.baseline, args.candidates, args.output)


if __name__ == "__main__":
    main()
