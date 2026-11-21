"""保存した OOF から最終モデルを固定し、カード群別の誤差を集計する。"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from math import isclose
from pathlib import Path

import numpy as np
from dataset import CARD_FILES, ID, TARGET, TRANSACTION_FILES, load_cards
from train import EXTREME, HYPOTHESES, ROOT, make_folds, rmse

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_regression
from kumite.experiments import ExperimentRecord, compare_runs
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


def select(root: Path, candidates: list[str], repeat: list[str] | None) -> dict[str, object]:
    data = root / "data"
    store = RunStore(root)
    train, _, _ = load_cards(data)
    y = train[TARGET].to_numpy()
    data_hash = files_sha256(data / name for name in (*CARD_FILES, *TRANSACTION_FILES))

    def scored(record: ExperimentRecord) -> tuple[np.ndarray, float]:
        run_id = record.run_id
        trial = run_id.split("-", 1)[0]
        if (
            record.competition_id != "015"
            or record.metric != "RMSE"
            or record.greater_is_better is not False
            or trial not in HYPOTHESES
            or record.params.get("trial") != trial
        ):
            raise ValueError(f"{run_id} の実験定義または評価指標が不正です")
        if (
            record.data_hash != data_hash
            or record.split_version != f"card-extreme-v1-seed{record.seed}"
            or record.split_hash != array_sha256(make_folds(y, record.seed))
        ):
            raise ValueError(f"{run_id} のデータまたは分割が一致しません")
        oof = store.load_array(run_id, "oof.npy")
        validate_regression(oof, len(y))
        score = rmse(y, oof)
        if not isclose(score, record.oof_score, rel_tol=1e-12):
            raise ValueError(f"{run_id} の記録済み OOF RMSE が再計算値と一致しません")
        return oof, score

    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("重複のない比較 run を指定してください")
    baseline = candidates[0]
    records = {run_id: store.record(run_id) for run_id in candidates}
    base = records[baseline]
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for run_id, record in records.items():
        if record.seed != base.seed:
            raise ValueError("比較 run の seed が一致しません")
        compare_runs(base, record)
        predictions[run_id], scores[run_id] = scored(record)
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
            if a.seed == base.seed or a.seed != b.seed:
                raise ValueError("同じ比較ペアを別 seed で再比較してください")
            compare_runs(a, b)
            repeated = {}
            for original, rerun in ((base, a), (records[winner], b)):
                if (original.params, original.data_version, original.model) != (
                    rerun.params,
                    rerun.data_version,
                    rerun.model,
                ):
                    raise ValueError(f"{rerun.run_id} は seed 以外の実験設定が異なります")
                _, repeated[rerun.run_id] = scored(rerun)
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
    submission = store.artifact_path(final, "submission.csv")
    decision.update(
        generated_at_utc=datetime.now(UTC).isoformat(),
        baseline=baseline,
        compared_runs=candidates,
        data_hash=data_hash,
        split_version=base.split_version,
        split_hash=base.split_hash,
        seed=base.seed,
        submission=submission.relative_to(root).as_posix(),
        submission_sha256=records[final].artifacts["submission.csv"],
        rule="OOF RMSE 最小を候補とする。対基準 runの差の95%区間がゼロを含む場合、別 seed でも改善したときだけ候補を採用する。",
    )
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "selection.json")
    args = parser.parse_args()
    result = select(ROOT, args.runs, args.repeat)
    if "final_run" in result:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
