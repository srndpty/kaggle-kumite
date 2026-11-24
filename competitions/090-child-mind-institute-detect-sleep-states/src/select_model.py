"""保存 OOF を再採点し、外部評価前の選定を固定する。"""

import json

import numpy as np
import polars as pl
from metric import confidence_order, matches, score
from pipeline import EVENTS
from train import ROOT, SEED, SPLIT, TRIALS

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore


def errors(truth: pl.DataFrame, oof: pl.DataFrame) -> dict[str, dict[str, int]]:
    truth = truth.drop_nulls("step")
    totals: dict[str, dict[str, int]] = {}
    for event in EVENTS:
        actual = truth.filter(pl.col("event") == event)
        predicted = oof.filter(pl.col("event") == event)
        series_ids = truth["series_id"].unique().to_list()
        groups = actual.partition_by("series_id", as_dict=True)
        proposals = list(
            predicted.filter(pl.col("series_id").is_in(series_ids))
            .partition_by("series_id", as_dict=True)
            .items()
        )
        for tolerance in (12, 360):
            tp = duplicate = false_positive = 0
            for key, part in proposals:
                observed = groups.get(key)
                steps = (
                    observed["step"].sort().to_numpy()
                    if observed is not None
                    else np.empty(0, dtype=np.int64)
                )
                ordered = part.sort("step")
                confidence = ordered["score"].to_numpy()
                order = confidence_order(confidence)
                proposed = ordered["step"].to_numpy()[order]
                found = matches(steps, proposed, tolerance)
                lo = np.searchsorted(steps, proposed - tolerance + 1)
                hi = np.searchsorted(steps, proposed + tolerance)
                tp += int(found.sum())
                duplicate += int((~found & (hi > lo)).sum())
                false_positive += int((~found).sum())
            totals[f"{event}/{tolerance}"] = {
                "matched": tp,
                "missed": len(actual) - tp,
                "false_positive": false_positive,
                "duplicate": duplicate,
            }
    return totals


def validate_main_run(record: ExperimentRecord, data_hash: str, split_hash: str) -> None:
    if (
        record.competition_id != ROOT.name
        or record.metric != "event_detection_ap"
        or record.greater_is_better is not True
        or record.split_version != SPLIT
        or record.seed != SEED
        or (record.data_hash, record.split_hash) != (data_hash, split_hash)
        or len(record.fold_scores) != 3
    ):
        raise ValueError(f"{record.run_id}: incompatible main evaluation")


def validate_candidates(run_ids: list[str]) -> None:
    if len(set(run_ids)) != len(run_ids) or set(run_ids) != TRIALS.keys():
        raise ValueError("Selection requires E00 through E04 exactly once")


def select(run_ids: list[str]) -> None:
    validate_candidates(run_ids)
    store = RunStore(ROOT)
    baseline = store.record("E00")
    records = [store.record(run_id) for run_id in run_ids]
    for record in records:
        validate_main_run(record, baseline.data_hash, baseline.split_hash)
    truth = pl.read_csv(ROOT / "data/train_events.csv", schema_overrides={"series_id": pl.String})
    comparisons = []
    for record in records:
        run_id = record.run_id
        oof = pl.read_csv(
            store.artifact_path(run_id, "oof.csv"), schema_overrides={"series_id": pl.String}
        )
        detail = score(truth, oof)
        if not np.isclose(np.mean(list(detail.values())), record.oof_score, rtol=0, atol=1e-12):
            raise ValueError(f"{run_id}: OOF score mismatch")
        comparisons.append(
            {
                "run_id": run_id,
                "oof_score": record.oof_score,
                "fold_scores": record.fold_scores,
                "ap": detail,
                "errors": errors(truth, oof),
                "oof_sha256": record.artifacts["oof.csv"],
            }
        )
    winner = min(comparisons, key=lambda row: (-row["oof_score"], row["run_id"]))
    payload = {
        "selected_run": winner["run_id"],
        "data_hash": baseline.data_hash,
        "split_hash": baseline.split_hash,
        "comparisons": comparisons,
    }
    with (ROOT / "selection.json").open("x", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(winner["run_id"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    select(args.run_ids)
