"""同一 fold の OOF を比較し、提出候補と欠損別の損失を記録する。"""

import json

import numpy as np
from train import ROOT, load

from kumite.bootstrap import group_bootstrap_diff


def main() -> None:
    x, y, _, _ = load(ROOT / "data")
    runs = ROOT / "outputs" / "runs"
    roles = np.load(runs / "E00" / "split_roles.npy")
    groups = np.arange(len(y))
    missing = x.isna().any(axis=1).to_numpy()
    cat_missing = x.select_dtypes(exclude="number").isna().any(axis=1).to_numpy()
    results = {}

    def losses(run_id: str) -> np.ndarray:
        path = runs / run_id
        if not np.array_equal(roles, np.load(path / "split_roles.npy")):
            raise ValueError(f"{run_id} の fold が E00 と異なります")
        probability = np.load(path / "oof.npy")
        return -np.log(np.clip(np.where(y == 1, probability, 1 - probability), 1e-15, 1))

    baseline = losses("E00")
    selection = "E00"
    best = 0.0
    for run_id in ("E00", "E01", "E02", "E03", "E04"):
        loss = losses(run_id)
        diff = group_bootstrap_diff(baseline, loss, groups, transform=np.asarray)
        if diff.high < 0 and diff.point < best:
            best = diff.point
            selection = run_id
        results[run_id] = {
            "log_loss": float(loss.mean()),
            "delta": diff.point,
            "ci95": [diff.low, diff.high],
            "with_missing": float(loss[missing].mean()),
            "without_missing": float(loss[~missing].mean()),
            "with_categorical_missing": float(loss[cat_missing].mean()),
        }
    report = {"selected": selection, "runs": results}
    (ROOT / "outputs" / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
