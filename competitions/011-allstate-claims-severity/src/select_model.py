"""固定 CV の OOF を比較し、外部採点前に最終 run を決める。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dummy import ROOT
from train import FILES, SEED, groups, load, mae, make_splits

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_regression
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

CANDIDATES = ("E00", "E01", "E02", "E03", "E04")


def select(run_ids: list[str]) -> dict:
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("選定候補が重複しています")
    if set(run_ids) != set(CANDIDATES):
        raise ValueError("選定候補には E00〜E04 が必要です")
    data = ROOT / "data"
    train, _, _ = load(data)
    x = train.drop(columns=["id", "loss"])
    y = train["loss"].to_numpy(dtype=np.float64)
    version, splits = make_splits(x)
    split_hash = array_sha256(split_roles(len(y), splits))
    data_hash = files_sha256(data / name for name in FILES)
    store = RunStore(ROOT)
    predictions = {}
    scores = {}
    oof_hashes = {}
    for run_id in CANDIDATES:
        record = store.record(run_id)
        if (record.competition_id, record.metric, record.greater_is_better, record.seed) != (
            "011",
            "MAE",
            False,
            SEED,
        ) or record.params.get("variant") != run_id:
            raise ValueError(f"{run_id} の実験設定が一致しません")
        if (record.data_hash, record.split_version, record.split_hash) != (
            data_hash,
            version,
            split_hash,
        ):
            raise ValueError(f"{run_id} のデータまたは分割が一致しません")
        pred = store.load_array(run_id, "oof.npy")
        try:
            validate_regression(pred, len(y), lower=0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{run_id} の OOF が不正です: {exc}") from exc
        score = mae(y, pred)
        if not np.isfinite(score) or not np.isclose(score, record.oof_score, rtol=0, atol=1e-9):
            raise ValueError(f"{run_id} の OOF MAE が実験ログと一致しません")
        for name in record.artifacts.keys() - {"oof.npy"}:
            store.artifact_path(run_id, name)
        predictions[run_id] = pred
        scores[run_id] = score
        oof_hashes[run_id] = record.artifacts["oof.npy"]
    baseline = predictions["E00"]
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
        "data_hash": data_hash,
        "split_version": version,
        "split_hash": split_hash,
        "seed": SEED,
        "oof_sha256": oof_hashes,
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
    parser.add_argument("--output", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"選定記録が既にあります: {args.output}")
        result = select(args.runs)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, LookupError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
