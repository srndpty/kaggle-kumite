"""選定済み OOF の群別診断と質問群単位のペア bootstrap。"""

import argparse
import json
from pathlib import Path

import numpy as np
from groups import question_groups
from quest import correlations, read_data, score

from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--compare")
    args = parser.parse_args()
    train, _, sample = read_data(ROOT / "data")
    columns = sample.columns[1:]
    y = train[columns].to_numpy(dtype=float)
    store = RunStore(ROOT)
    pred = store.load_array(args.run, "oof.npy")
    lengths = train["question_body"].fillna("").str.len().to_numpy()
    lower, upper = np.quantile(lengths, [1 / 3, 2 / 3])
    hosts = train["host"].value_counts().head(5).index
    metrics = {
        "targets": dict(zip(columns, correlations(y, pred).tolist(), strict=True)),
        "hosts": {
            str(host): score(y[train["host"] == host], pred[train["host"] == host])
            for host in hosts
        },
        "body_length": {
            label: {"rows": int(mask.sum()), "score": score(y[mask], pred[mask])}
            for label, mask in (
                ("short", lengths <= lower),
                ("medium", (lengths > lower) & (lengths <= upper)),
                ("long", lengths > upper),
            )
        },
        "body_length_cutoffs": [float(lower), float(upper)],
    }
    print(json.dumps(metrics, indent=2))
    if args.compare:
        alternative = store.load_array(args.compare, "oof.npy")
        groups = question_groups(train)
        order = np.argsort(groups)
        boundaries = np.flatnonzero(np.diff(groups[order])) + 1
        rows = np.split(order, boundaries)
        rng = np.random.default_rng(42)
        deltas = []
        for _ in range(400):
            sample_rows = np.concatenate(
                [rows[index] for index in rng.integers(len(rows), size=len(rows))]
            )
            deltas.append(
                score(y[sample_rows], pred[sample_rows])
                - score(y[sample_rows], alternative[sample_rows])
            )
        print("Paired bootstrap, 95%:", np.quantile(deltas, [0.025, 0.975]).tolist())


if __name__ == "__main__":
    main()
