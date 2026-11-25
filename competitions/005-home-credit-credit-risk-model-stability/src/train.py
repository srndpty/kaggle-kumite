"""週単位の CV と提出用の全行再学習。"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from pipeline import PARAMS, TABLES, WEEK, fit_predict, fold_weeks, load, shards, stability

from kumite.checks import validate_probabilities
from kumite.experiments import ExperimentRecord, compare_runs
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SLUG = "home-credit-credit-risk-model-stability"
TRIALS = {
    "E00": ("静的特徴で将来週の識別力を測れる", "静的特徴のみ"),
    "E01": ("過去申込の特徴で平均週別 Gini が改善する", "過去申込の件数・経過日数"),
    "E02": ("最近の行を重くすると Gini の低下が小さくなる", "学習重みの半減期26週"),
    "E03": ("欠損率が変化する列を除くと週別のばらつきが小さくなる", "欠損率差による列除外"),
    "E04": ("過去申込の最大金額が識別力を補う", "過去申込の融資額・カード限度額の最大値"),
}


def input_hash(data: Path, part: str) -> str:
    return files_sha256(path for table in TABLES for path in shards(data, part, table))


def run(data: Path, trial: str, run_id: str) -> None:
    started = perf_counter()
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        # 全試行で同じ入力一式をハッシュする。
        digest = input_hash(data, "train")
        version = code_version(REPO_ROOT)
        frame = load(data, "train", history=trial != "E00", max_amount=trial == "E04")
        weeks, target = frame[WEEK].to_numpy(), frame["target"].to_numpy()
        splits = fold_weeks(weeks, target)
        fold = np.full(len(frame), -1, dtype=np.int8)
        oof = np.full(len(frame), np.nan)
        scores = []
        for index, (train, valid) in enumerate(splits):
            fold[valid] = index
            pred = fit_predict(frame[train], frame[valid], trial)
            validate_probabilities(np.column_stack((1 - pred, pred)), len(valid), 2)
            oof[valid] = pred
            scores.append(stability(weeks[valid], target[valid], oof[valid]))
            print(f"{run_id} fold {index}: {scores[-1]}", flush=True)
        valid = fold >= 0
        pooled = stability(weeks[valid], target[valid], oof[valid])
        fold_scores = [summary["score"] for summary in scores]
        hypothesis, change = TRIALS[trial]
        record = ExperimentRecord(
            competition_id="005",
            run_id=run_id,
            metric="mean_window_gini_stability",
            greater_is_better=True,
            data_version=trial,
            data_hash=digest,
            split_version="week-expanding-v1",
            split_hash=array_sha256(fold),
            seed=42,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model="LightGBM",
            params=PARAMS,
            fold_scores=fold_scores,
            oof_score=float(np.mean(fold_scores)),
            runtime_sec=perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
        )

        def save(output: Path) -> None:
            np.save(output / "oof.npy", oof)
            np.save(output / "fold.npy", fold)
            (output / "metrics.json").write_text(
                json.dumps({"folds": scores, "pooled": pooled}, indent=2), encoding="utf-8"
            )

        print(store.publish(reservation, record, save), flush=True)


def write_notebook(trial: str, owner: str, output: Path) -> None:
    with_history = trial != "E00"
    source = Path(__file__).with_name("pipeline.py").read_text(encoding="utf-8")
    source += f"""

test_base, = Path('/kaggle/input').rglob('test_base.parquet')
data = test_base.parents[2]
train = load(data, 'train', history={with_history!r}, max_amount={trial == "E04"!r})
test = load(data, 'test', history={with_history!r}, max_amount={trial == "E04"!r})
if test.is_empty() or not test[ID].equals(pl.read_parquet(test_base, columns=[ID])[ID]):
    raise ValueError('公式テストの case_id 順が一致しません')
prediction = fit_predict(train, test, {trial!r})
if (prediction.shape != (len(test),) or not np.isfinite(prediction).all()
        or np.any((prediction < 0) | (prediction > 1))):
    raise ValueError('提出確率が不正です')
test.select(ID).with_columns(pl.Series('score', prediction)).write_csv(
    '/kaggle/working/submission.csv'
)
"""
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/kumite-home-credit-stability-{trial.lower()}",
        title=f"Kumite Home Credit Stability {trial}",
        competition=SLUG,
    )


def export(data: Path, trial: str, owner: str, run_id: str, source_run: str) -> None:
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        selected = store.record(source_run)
        if selected.data_version != trial:
            raise ValueError("再学習する構成と CV run が一致しません")
        if input_hash(data, "train") != selected.data_hash:
            raise ValueError("再学習するデータのハッシュが CV run と一致しません")
        record = replace(
            selected,
            run_id=run_id,
            hypothesis="CV で評価済みの構成を全学習行で再学習する",
            changed_one_thing="学習に全学習行を使う",
            fold_scores=[],
            late_score=None,
            runtime_sec=None,
            peak_memory_mb=None,
            code_version=code_version(REPO_ROOT),
            inputs={
                **store.input_hashes(source_run, "oof.npy", "fold.npy", "metrics.json"),
                "test_parquet": input_hash(data, "test"),
            },
            conclusion="oof_score は選定した CV run の参照値",
        )
        print(store.publish(reservation, record, lambda out: write_notebook(trial, owner, out)))


def select_final(store: RunStore) -> ExperimentRecord:
    candidates = [store.record(trial) for trial in TRIALS]
    baseline = candidates[0]
    for candidate in candidates:
        compare_runs(baseline, candidate)
        if len(candidate.fold_scores) != 3 or not np.isfinite(candidate.oof_score):
            raise ValueError(f"{candidate.run_id} の CV 結果が不完全です")
    return max(candidates, key=lambda candidate: candidate.oof_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=(*TRIALS, "best"))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--run-id")
    parser.add_argument("--owner", help="指定すると提出 Notebook を生成する")
    parser.add_argument("--source-run", help="全行再学習で参照する CV run")
    args = parser.parse_args()
    if args.trial == "best":
        if not args.owner or args.source_run:
            parser.error("best には --owner が必要です。--source-run は使えません")
        selected = select_final(RunStore(ROOT))
        print(f"選定: {selected.run_id} {selected.oof_score:.8f}", flush=True)
        export(args.data, selected.data_version, args.owner, args.run_id or "F00", selected.run_id)
    elif args.owner:
        export(
            args.data,
            args.trial,
            args.owner,
            args.run_id or f"F-{args.trial}",
            args.source_run or args.trial,
        )
    else:
        run(args.data, args.trial, args.run_id or args.trial)


if __name__ == "__main__":
    main()
