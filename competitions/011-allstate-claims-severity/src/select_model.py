"""固定 CV の OOF を比較し、外部採点前に最終 run を決める。"""

import argparse
import json
from dataclasses import asdict

import numpy as np
from dummy import ROOT
from train import FILES, SEED, groups, load, mae, make_splits

from kumite.bootstrap import group_bootstrap_diff
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


def select(run_ids: list[str]) -> dict:
    data = ROOT / "data"
    train, _, _ = load(data)
    x = train.drop(columns=["id", "loss"])
    y = train["loss"].to_numpy(dtype=np.float64)
    version, splits = make_splits(x)
    split_hash = array_sha256(split_roles(len(y), splits))
    data_hash = files_sha256(data / name for name in FILES)
    store = RunStore(ROOT)
    predictions = {}
    for run_id in run_ids:
        record = store.record(run_id)
        if (record.data_hash, record.split_version, record.split_hash) != (
            data_hash,
            version,
            split_hash,
        ):
            raise ValueError(f"{run_id} のデータまたは分割が一致しません")
        predictions[run_id] = store.load_array(run_id, "oof.npy")
    baseline = predictions["E00"]
    scores = {run_id: mae(y, pred) for run_id, pred in predictions.items()}
    best = min(scores, key=lambda run_id: (scores[run_id], run_id))
    difference = group_bootstrap_diff(
        np.abs(y - baseline),
        np.abs(y - predictions[best]),
        groups(x),
        transform=lambda value: value,
        seed=SEED,
    )
    q25, q75, q99 = np.quantile(y, [0.25, 0.75, 0.99])
    bands = {
        "bottom_quartile": y < q25,
        "middle_half": (y >= q25) & (y < q75),
        "upper_except_top_1pct": (y >= q75) & (y < q99),
        "top_1pct": y >= q99,
    }
    return {
        "selected": best if difference.high < 0 else "E00",
        "lowest_oof": best,
        "best_minus_e00_bootstrap": asdict(difference),
        "oof_mae": scores,
        "bands": {
            name: {
                run_id: {
                    "mae": mae(y[mask], pred[mask]),
                    "signed_error": float(np.mean(pred[mask] - y[mask])),
                    "overprediction": float(np.maximum(pred[mask] - y[mask], 0).mean()),
                }
                for run_id, pred in predictions.items()
            }
            for name, mask in bands.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(select(args.runs), indent=2))


if __name__ == "__main__":
    main()
