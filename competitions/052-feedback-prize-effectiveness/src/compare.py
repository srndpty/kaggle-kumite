"""保存 OOF でエッセイ単位のペア比較を行う。"""

import argparse
import json
from dataclasses import asdict

import numpy as np
from model import CLASSES, ROLE, TEXT
from train import COMP, assess_for, load_cv, logloss_rows

from kumite.bootstrap import group_bootstrap_diff
from kumite.quality_gates import verify_cv_run
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    frame, labels, groups, splits, evaluation = load_cv()
    assess = assess_for(labels, splits, evaluation)
    store = RunStore(COMP)

    def losses(run_id: str) -> np.ndarray:
        verify_cv_run(store, run_id, assess)
        return logloss_rows(labels, store.load_array(run_id, "oof.npy"))

    baseline = losses(args.baseline)
    roles = frame[ROLE].to_numpy()
    short = np.asarray([len(text.split()) <= 16 for text in frame[TEXT]], dtype=np.bool_)
    masks: dict[str, np.ndarray] = {
        **{str(role): roles == role for role in sorted(set(roles))},
        **{label: labels == i for i, label in enumerate(CLASSES)},
        "short": short,
        "longer": ~short,
    }
    for run_id in args.candidates:
        current = losses(run_id)
        edge = current - baseline
        diff = group_bootstrap_diff(baseline, current, groups, transform=lambda values: values)
        print(
            json.dumps(
                {
                    "run": run_id,
                    "logloss": float(current.mean()),
                    "delta": asdict(diff),
                    "slices": {
                        name: {"rows": int(mask.sum()), "delta": float(edge[mask].mean())}
                        for name, mask in masks.items()
                        if mask.any()
                    },
                }
            )
        )


if __name__ == "__main__":
    main()
