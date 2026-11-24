"""グループ分割の OOF だけで提出モデルを固定する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline import EVENTS
from prepare import ROOT, SEED

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore

SMALL_DIFF = 0.005
RESAMPLES = 200


def bootstrap_ap(
    labels: np.ndarray, predictions: np.ndarray, subject_codes: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    values = np.zeros(len(samples))
    for actual, probabilities in zip(labels.T, predictions.T, strict=True):
        order = np.argsort(-probabilities)
        ordered = probabilities[order]
        edges = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        codes = subject_codes[order]
        positives = actual[order].astype(bool)
        for index, draws in enumerate(samples):
            weights = draws[codes]
            all_by_score = np.add.reduceat(weights, edges, dtype=np.int64)
            pos_by_score = np.add.reduceat(weights * positives, edges, dtype=np.int64)
            true_pos = pos_by_score.cumsum()
            total = all_by_score.cumsum()
            if true_pos[-1] > 0:
                active = pos_by_score > 0
                values[index] += (
                    true_pos[active] / total[active] @ pos_by_score[active]
                ) / true_pos[-1]
    return values / len(EVENTS)


def simpler(record: ExperimentRecord) -> int:
    return int(str(record.params["trial"]).removeprefix("E"))


def select(store: RunStore, run_ids: list[str]) -> dict[str, object]:
    candidates = sorted(
        (store.record(run_id) for run_id in run_ids),
        key=lambda record: (-record.oof_score, simpler(record), record.run_id),
    )
    best, runner_up = candidates[:2]
    if any(
        (record.data_hash, record.split_hash, record.split_version, record.metric)
        != (best.data_hash, best.split_hash, best.split_version, best.metric)
        for record in candidates
    ):
        raise ValueError("Model candidates must use the same data, folds and metric")
    outcome: dict[str, object] = {"run_id": best.run_id, "runner_up": runner_up.run_id}
    if best.oof_score - runner_up.oof_score <= SMALL_DIFF:
        split = pd.read_csv(store.artifact_path(best.run_id, "split.csv"), dtype={"subject": str})
        labels = np.load(store.artifact_path(best.run_id, "labels.npy"), mmap_mode="r")
        other_labels = np.load(store.artifact_path(runner_up.run_id, "labels.npy"), mmap_mode="r")
        if not np.array_equal(labels, other_labels):
            raise ValueError("The OOF label artifacts do not align")
        codes, subjects = pd.factorize(split["subject"])
        if (codes < 0).any():
            raise ValueError("Missing subject IDs")
        row_subjects = np.repeat(codes.astype(np.int16), split["annotated"].to_numpy())
        rng = np.random.default_rng(SEED)
        draws = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), RESAMPLES)
        samples = draws.astype(np.int16)
        scores = [
            bootstrap_ap(
                labels,
                np.load(store.artifact_path(record.run_id, "oof.npy"), mmap_mode="r"),
                row_subjects,
                samples,
            )
            for record in (best, runner_up)
        ]
        interval = np.quantile(scores[0] - scores[1], [0.025, 0.975]).tolist()
        outcome["best_minus_runner_up_95_interval"] = interval
        if interval[0] <= 0 <= interval[1]:
            outcome["run_id"] = min((best, runner_up), key=simpler).run_id
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/selection.json")
    args = parser.parse_args()
    if len(set(args.run_ids)) != len(args.run_ids) or len(args.run_ids) < 2:
        parser.error("Provide at least two distinct run IDs")
    result = json.dumps(select(RunStore(ROOT), args.run_ids), indent=2)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
