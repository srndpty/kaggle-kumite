"""画像単位のペア bootstrap で最終 run を選ぶ。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from train import ROOT

from kumite.bootstrap import group_bootstrap_diff
from kumite.runs import RunStore


def read_oof(store: RunStore, run_id: str) -> pd.DataFrame:
    table = pd.read_csv(store.artifact_path(run_id, "oof.csv"))
    table = table.set_index(["image_id", "class_id"]).sort_index()
    valid_classes = table.index.get_level_values("class_id").isin([1, 2, 3, 4]).all()
    if (
        not table.index.is_unique
        or not valid_classes
        or not table.groupby(level="image_id").size().eq(4).all()
    ):
        raise ValueError(f"{run_id} に4クラス揃っていない画像があります")
    return table


def select(store: RunStore, baseline: str, candidates: list[str], output: Path) -> str:
    base = store.record(baseline)
    base_table = read_oof(store, baseline)
    records = [store.record(run_id) for run_id in candidates]
    for candidate in records:
        if (candidate.data_hash, candidate.split_hash, candidate.metric) != (
            base.data_hash,
            base.split_hash,
            base.metric,
        ):
            raise ValueError(f"{candidate.run_id} は基準と評価条件が揃っていません")
    best = max([base, *records], key=lambda record: record.oof_score)
    best_table = read_oof(store, best.run_id)
    if not base_table.index.equals(best_table.index) or not base_table["fold"].equals(
        best_table["fold"]
    ):
        raise ValueError("OOF の画像・クラスまたは fold が揃っていません")
    delta = group_bootstrap_diff(
        base_table["dice"].to_numpy(),
        best_table["dice"].to_numpy(),
        base_table.index.get_level_values("image_id").to_numpy(),
        transform=lambda value: value,
    )
    chosen = best.run_id if delta.low > 0 else baseline
    evidence = {
        "baseline": baseline,
        "candidates": {record.run_id: record.oof_score for record in [base, *records]},
        "highest_oof": best.run_id,
        "paired_dice_difference": asdict(delta),
        "selected_run": chosen,
        "oof_hashes": {record.run_id: record.artifacts["oof.csv"] for record in [base, *records]},
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(select(RunStore(ROOT), args.baseline, args.candidates, args.output))


if __name__ == "__main__":
    main()
