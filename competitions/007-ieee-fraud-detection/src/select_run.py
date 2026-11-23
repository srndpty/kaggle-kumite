"""CV 記録から最終候補を固定する。"""

import argparse
import json
from pathlib import Path

import numpy as np

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SMALL_DELTA = 0.002
COMPLEXITY = {"base": 0, "relative_amount": 1, "clock": 2, "history": 2, "identity": 42}


def selection(store: RunStore, run_ids: list[str]) -> dict[str, object]:
    records = [store.record(run_id) for run_id in run_ids]
    if not records or len(set(run_ids)) != len(run_ids):
        raise ValueError("候補に重複があるか、候補がありません")
    first = records[0]
    comparisons: dict[tuple[int, str], ExperimentRecord] = {}
    for record in records:
        if (
            (record.data_hash, record.split_hash, record.split_version)
            != (first.data_hash, first.split_hash, first.split_version)
            or record.metric != "ROC AUC"
            or len(record.fold_scores) != 3
            or "oof.npy" not in record.artifacts
        ):
            raise ValueError(f"{record.run_id} は比較可能な CV run ではありません")
        store.load_array(record.run_id, "oof.npy")
        roles = store.load_array(record.run_id, "split_roles.npy")
        if roles.ndim != 2 or roles.shape[0] != 3 or array_sha256(roles) != record.split_hash:
            raise ValueError(f"{record.run_id} の fold 成果物が不正です")
        key = (record.seed, str(record.params["trial"]))
        if key in comparisons:
            raise ValueError("同じ seed と特徴設定の候補が複数あります")
        comparisons[key] = record

    def score(record: ExperimentRecord) -> float:
        return float(np.mean(record.fold_scores))

    base = comparisons[42, "base"]
    eligible = [base]
    excluded = {}
    for (seed, trial), record in comparisons.items():
        if seed != 42 or trial == "base":
            continue
        delta = score(record) - score(base)
        if delta <= 0:
            excluded[record.run_id] = "基準より改善していない"
            continue
        if delta <= SMALL_DELTA:
            check_base = comparisons.get((43, "base"))
            check_trial = comparisons.get((43, trial))
            if check_base is None or check_trial is None:
                raise ValueError(f"{record.run_id} は seed 43 のペアが必要です")
            if score(check_trial) <= score(check_base):
                excluded[record.run_id] = "seed 43 で改善しない"
                continue
        eligible.append(record)

    winner = min(
        eligible,
        key=lambda record: (
            -score(record),
            COMPLEXITY[str(record.params["trial"])],
            record.run_id,
        ),
    )
    return {
        "selected": winner.run_id,
        "data_hash": first.data_hash,
        "split_hash": first.split_hash,
        "split_version": first.split_version,
        "small_delta": SMALL_DELTA,
        "excluded": excluded,
        "candidates": {
            record.run_id: {
                "mean_auc": score(record),
                "seed": record.seed,
                "oof_sha256": record.artifacts["oof.npy"],
            }
            for record in records
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    result = selection(RunStore(ROOT), args.runs)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(result["selected"])


if __name__ == "__main__":
    main()
