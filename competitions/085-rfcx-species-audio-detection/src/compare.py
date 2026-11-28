"""再採点済み OOF の録音単位ペア bootstrap と内訳を出す。"""

import argparse
import json

import numpy as np
from dataset import RATE
from train import COMP, SEED, load_cv

from kumite.lwlrap import lwlrap, positive_precisions
from kumite.quality_gates import dense_oof, verify_cv_run
from kumite.runs import RunStore


def compare() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    _, keys, _, _, true, _, labels, splits, evaluation = load_cv()
    store = RunStore(COMP)
    assess = dense_oof(
        labels, splits, lwlrap, expected_shape=labels.shape, data_hash=evaluation.data_hash
    )

    def verified(run_id: str) -> np.ndarray:
        verify_cv_run(store, run_id, assess)
        return positive_precisions(labels, store.load_array(run_id, "oof.npy"))

    baseline = verified(args.baseline)
    counts = labels.sum(axis=1)
    frequency = labels.sum(axis=0)
    short = np.zeros_like(labels)
    for row, key in enumerate(keys):
        for species in np.flatnonzero(labels[row]):
            short[row, species] = (
                max(e.end - e.start for e in true[key] if e.species == species) <= RATE
            )
    rare = frequency <= np.median(frequency)
    masks = {
        "rare": labels & rare,
        "frequent": labels & ~rare,
        "short_at_most_1s": labels & short,
        "long_over_1s": labels & ~short,
    }
    for run_id in args.candidates:
        values = verified(run_id)
        diff = values - baseline
        totals = diff.sum(axis=1)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(2000):
            rows = rng.integers(len(labels), size=len(labels))
            denom = int(counts[rows].sum())
            if denom:
                draws.append(float(totals[rows].sum() / denom))
        intervals = np.quantile(draws, [0.025, 0.975]).tolist()
        print(
            json.dumps(
                {
                    "run": run_id,
                    "lwlrap": float(values.sum() / counts.sum()),
                    "difference": float(diff.sum() / counts.sum()),
                    "interval": intervals,
                    "slices": {
                        name: {"positives": int(mask.sum()), "delta": float(diff[mask].mean())}
                        for name, mask in masks.items()
                        if mask.any()
                    },
                }
            )
        )


if __name__ == "__main__":
    compare()
