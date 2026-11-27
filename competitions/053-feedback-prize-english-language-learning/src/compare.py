"""基準 run に対する MCRMSE 差のグループ bootstrap を表示する。"""

import argparse

import numpy as np
from train import COMP, SEED, load_cv, mcrmse

from kumite.quality_gates import dense_oof, verify_cv_run
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    _, labels, groups, splits, evaluation = load_cv()
    assess = dense_oof(
        labels, splits, mcrmse, expected_shape=labels.shape, data_hash=evaluation.data_hash
    )
    store = RunStore(COMP)
    _, group_ids = np.unique(groups, return_inverse=True)
    group_ids = group_ids.reshape(-1)
    counts = np.bincount(group_ids)

    def loss_by_group(run_id: str) -> np.ndarray:
        verify_cv_run(store, run_id, assess)
        squared = np.square(labels - store.load_array(run_id, "oof.npy"))
        totals = np.zeros((len(counts), labels.shape[1]))
        np.add.at(totals, group_ids, squared)
        return totals

    baseline = loss_by_group(args.baseline)
    for run_id in args.candidates:
        candidate = loss_by_group(run_id)
        rng = np.random.default_rng(SEED)
        deltas = []
        for _ in range(2000):
            sampled = rng.integers(len(counts), size=len(counts))
            rows = counts[sampled].sum()
            base_score = np.sqrt(baseline[sampled].sum(axis=0) / rows).mean()
            candidate_score = np.sqrt(candidate[sampled].sum(axis=0) / rows).mean()
            deltas.append(candidate_score - base_score)
        low, high = np.quantile(deltas, [0.025, 0.975])
        rmse = np.sqrt(candidate.sum(axis=0) / counts.sum())
        print(f"{run_id}: {rmse.mean():.6f} [{low:+.6f}, {high:+.6f}] {rmse.round(6)}")


if __name__ == "__main__":
    main()
