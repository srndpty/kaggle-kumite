"""保存した OOF から最終モデルを固定し、カード群別の誤差を集計する。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dataset import CARD_FILES, ID, TARGET, TRANSACTION_FILES, load_cards
from train import EXTREME, ROOT, make_folds, rmse

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_regression
from kumite.experiments import compare_runs
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


def select(root: Path, candidates: list[str], repeat: list[str] | None) -> dict[str, object]:
    data = root / "data"
    store = RunStore(root)
    train, _, _ = load_cards(data)
    y = train[TARGET].to_numpy()
    data_hash = files_sha256(data / name for name in (*CARD_FILES, *TRANSACTION_FILES))

    def scored(run_id: str) -> np.ndarray:
        record = store.record(run_id)
        if record.data_hash != data_hash or record.split_hash != array_sha256(
            make_folds(y, record.seed)
        ):
            raise ValueError(f"{run_id} のデータまたは分割が一致しません")
        oof = store.load_array(run_id, "oof.npy")
        validate_regression(oof, len(y))
        return oof

    baseline = candidates[0]
    records = [store.record(run_id) for run_id in candidates]
    for record in records[1:]:
        compare_runs(records[0], record)
    predictions = {run_id: scored(run_id) for run_id in candidates}
    scores = {run_id: rmse(y, oof) for run_id, oof in predictions.items()}
    winner = min(scores, key=scores.__getitem__)
    base_loss = np.square(y - predictions[baseline])
    loss = np.square(y - predictions[winner])
    decision: dict[str, object] = {"scores": scores, "candidate": winner}
    final = winner
    if winner != baseline:
        interval = group_bootstrap_diff(base_loss, loss, train[ID].to_numpy())
        decision["bootstrap"] = asdict(interval)
        if interval.high >= 0:
            if repeat is None:
                decision["repeat_required"] = [baseline, winner]
                return decision
            repeat_base, repeat_candidate = repeat
            a, b = store.record(repeat_base), store.record(repeat_candidate)
            if (
                a.seed == records[0].seed
                or a.seed != b.seed
                or a.params["trial"] != store.record(baseline).params["trial"]
                or b.params["trial"] != store.record(winner).params["trial"]
            ):
                raise ValueError("別 seed で同じ二つの実験を比較してください")
            repeated = {run_id: rmse(y, scored(run_id)) for run_id in repeat}
            decision["repeat_scores"] = repeated
            if repeated[repeat_candidate] >= repeated[repeat_base]:
                final = baseline

    squared = np.square(y - predictions[final])
    total = float(squared.sum())

    def summarize(mask: np.ndarray) -> dict[str, int | float]:
        return {
            "count": int(mask.sum()),
            "rmse": rmse(y[mask], predictions[final][mask]),
            "squared_error_share": float(squared[mask].sum() / total) if total else 0.0,
        }

    years = train["first_active_month"].str[:4].fillna("欠損")
    decision["final_run"] = final
    decision["submission"] = str(store.artifact_path(final, "submission.csv"))
    decision["group_errors"] = {
        "extreme": summarize(y < EXTREME),
        "normal": summarize(y >= EXTREME),
        "first_active_year": {
            year: summarize((years == year).to_numpy()) for year in sorted(set(years))
        },
    }
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="基準 run、続けて比較対象")
    parser.add_argument("--repeat", nargs=2, help="別 seed の基準 run と候補 run")
    args = parser.parse_args()
    result = select(ROOT, args.runs, args.repeat)
    if "final_run" in result:
        with (ROOT / "outputs" / "selection.json").open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
