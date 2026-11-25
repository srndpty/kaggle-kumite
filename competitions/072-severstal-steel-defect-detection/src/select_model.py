"""画像単位のペア bootstrap で最終 run を選ぶ。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from cv_checks import evaluation_key
from train import ROOT

from kumite.bootstrap import group_bootstrap_diff
from kumite.runs import RunStore


def read_oof(store: RunStore, run_id: str) -> pd.DataFrame:
    # artifact_path が実ファイルの SHA-256 を実験ログと照合する。
    table = pd.read_csv(store.artifact_path(run_id, "oof.csv"), dtype={"image_id": str})
    if table.empty or table["image_id"].isna().any():
        raise ValueError(f"{run_id} の OOF 画像 ID が空です")
    table = table.set_index(["image_id", "class_id"]).sort_index()
    valid_classes = table.index.get_level_values("class_id").isin([1, 2, 3, 4]).all()
    if (
        not table.index.is_unique
        or not valid_classes
        or not table.groupby(level="image_id").size().eq(4).all()
    ):
        raise ValueError(f"{run_id} に4クラス揃っていない画像があります")
    folds = table["fold"]
    if not folds.isin([0, 1, 2]).all() or not folds.groupby(level="image_id").nunique().eq(1).all():
        raise ValueError(f"{run_id} の OOF fold が不正です")
    values = table["dice"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{run_id} の OOF Dice が不正です")
    if not np.isclose(values.mean(), store.record(run_id).oof_score, rtol=0, atol=1e-9):
        raise ValueError(f"{run_id} の OOF Dice が実験ログと一致しません")
    return table


def select(store: RunStore, baseline: str, candidates: list[str], output: Path) -> str:
    base = store.record(baseline)
    base_table = read_oof(store, baseline)
    records = [store.record(run_id) for run_id in candidates]
    if base.metric != "mean_dice" or not base.greater_is_better:
        raise ValueError("基準の評価指標は最大化する mean_dice が必要です")
    tables = {baseline: base_table}
    scores = {baseline: float(base_table["dice"].mean())}
    for candidate in records:
        if evaluation_key(candidate) != evaluation_key(base):
            raise ValueError(f"{candidate.run_id} は基準と評価条件が揃っていません")
        table = read_oof(store, candidate.run_id)
        if not base_table.index.equals(table.index) or not base_table["fold"].equals(table["fold"]):
            raise ValueError(f"{candidate.run_id} の OOF 画像・クラスまたは fold が揃っていません")
        tables[candidate.run_id] = table
        scores[candidate.run_id] = float(table["dice"].mean())
    best = max(scores, key=scores.__getitem__)
    best_table = tables[best]
    delta = group_bootstrap_diff(
        base_table["dice"].to_numpy(),
        best_table["dice"].to_numpy(),
        base_table.index.get_level_values("image_id").to_numpy(),
        transform=lambda value: value,
    )
    chosen = best if delta.low > 0 else baseline
    evidence = {
        "baseline": baseline,
        "candidates": scores,
        "highest_oof": best,
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
