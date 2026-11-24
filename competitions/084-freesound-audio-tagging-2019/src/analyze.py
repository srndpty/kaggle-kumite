"""curated OOF を再採点し、比較・選定の記録を出力する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from metric import lwlrap, positive_precisions
from train import ROOT

from kumite.crossfit import folds_to_splits
from kumite.runs import RunStore

CANDIDATES = ("E01", "E02", "E03", "E04", "E05")


def calculate_selection(store: RunStore, run_ids: list[str]) -> dict[str, object]:
    records = [store.record(run_id) for run_id in run_ids]
    if tuple(record.params.get("experiment") for record in records) != CANDIDATES:
        raise ValueError("Expected one complete candidate for each experiment E01 through E05")
    base = records[0]
    labels = store.load_array(base.run_id, "labels.npy")
    manifest = pd.read_csv(store.artifact_path(base.run_id, "manifest.csv"))
    lengths = manifest["seconds"].to_numpy()
    splits = folds_to_splits(manifest["fold"].to_numpy())
    tag_counts = labels.sum(axis=1)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(labels), (2000, len(labels)))
    denominator = tag_counts[draws].sum(axis=1)
    base_scores = store.load_array(base.run_id, "oof.npy")
    base_precision = positive_precisions(labels, base_scores)
    tag_frequency = labels.sum(axis=0)
    comparisons = []
    for record in records:
        if (
            record.params.get("benchmark")
            or record.metric != "lwlrap"
            or not record.greater_is_better
            or record.seed != base.seed
            or record.data_hash != base.data_hash
            or record.split_hash != base.split_hash
            or record.split_version != base.split_version
            or record.params.get("updates") != base.params.get("updates")
        ):
            raise ValueError(f"Run conditions differ: {record.run_id}")
        if record.params.get("tags") != base.params.get("tags") or not np.array_equal(
            store.load_array(record.run_id, "labels.npy"), labels
        ):
            raise ValueError(f"Label data differs: {record.run_id}")
        store.artifact_path(record.run_id, "manifest.csv")
        if record.artifacts["manifest.csv"] != base.artifacts["manifest.csv"]:
            raise ValueError(f"Manifest differs: {record.run_id}")
        actual = store.load_array(record.run_id, "oof.npy")
        score = lwlrap(labels, actual)
        if not np.isclose(score, record.oof_score, rtol=0, atol=1e-12):
            raise ValueError(f"Stored score mismatch: {record.run_id}")
        fold_scores = [lwlrap(labels[valid], actual[valid]) for _, valid in splits]
        if len(record.fold_scores) != len(fold_scores) or not np.allclose(
            fold_scores, record.fold_scores, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Stored fold scores mismatch: {record.run_id}")
        paired_precision = positive_precisions(labels, actual) - base_precision
        delta = paired_precision.sum(axis=1)
        interval = np.percentile(delta[draws].sum(axis=1) / denominator, [2.5, 97.5])
        slices = {}
        for name, mask in (
            ("length/<=2s", lengths <= 2),
            ("length/2-5s", (lengths > 2) & (lengths <= 5)),
            ("length/>5s", lengths > 5),
            ("tags/1", tag_counts == 1),
            ("tags/2+", tag_counts >= 2),
        ):
            if mask.any():
                slices[name] = {
                    "clips": int(mask.sum()),
                    "paired_delta": float(delta[mask].sum() / tag_counts[mask].sum()),
                }
        for name, mask in (
            ("frequency/1-50", (tag_frequency > 0) & (tag_frequency <= 50)),
            ("frequency/51-100", (tag_frequency > 50) & (tag_frequency <= 100)),
            ("frequency/101+", tag_frequency > 100),
        ):
            if mask.any():
                count = int(tag_frequency[mask].sum())
                slices[name] = {
                    "positive_tags": count,
                    "paired_delta": float(paired_precision[:, mask].sum() / count),
                }
        comparisons.append(
            {
                "run_id": record.run_id,
                "lwlrap": score,
                "delta_ci95": interval.tolist(),
                "slices": slices,
                "oof_sha256": record.artifacts["oof.npy"],
            }
        )
    winner = max(comparisons, key=lambda result: float(result["lwlrap"]))
    if float(winner["lwlrap"]) - base.oof_score < 0.005 and winner["delta_ci95"][0] <= 0:
        winner = comparisons[0]
    return {
        "selected_run": winner["run_id"],
        "data_hash": base.data_hash,
        "split_hash": base.split_hash,
        "comparisons": comparisons,
    }


def verify_selection(store: RunStore, choice: dict, requested: str) -> None:
    run_ids = [candidate["run_id"] for candidate in choice["comparisons"]]
    rebuilt = calculate_selection(store, run_ids)
    if rebuilt != choice or rebuilt["selected_run"] != requested:
        raise ValueError("Selection does not match the recomputed candidate ranking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    result = calculate_selection(RunStore(ROOT), args.runs)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
