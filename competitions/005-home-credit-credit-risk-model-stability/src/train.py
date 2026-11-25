"""週単位の CV と提出用の全行再学習。"""

import argparse
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import polars as pl
from pipeline import PARAMS, TABLES, WEEK, fit_predict, fold_weeks, load, shards, stability

from kumite.checks import validate_probabilities
from kumite.experiments import ExperimentRecord
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

METRIC = "mean_window_gini_stability"
SPLIT_VERSION = "week-expanding-v1"
PIPELINE = Path(__file__).with_name("pipeline.py")

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
        source = PIPELINE.read_bytes()
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
            metric=METRIC,
            greater_is_better=True,
            data_version=trial,
            data_hash=digest,
            split_version=SPLIT_VERSION,
            split_hash=array_sha256(fold),
            seed=PARAMS["seed"],
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

        if PIPELINE.read_bytes() != source:
            raise ValueError("CV 実行中に pipeline.py が変更されました")

        def save(output: Path) -> None:
            (output / "pipeline.py").write_bytes(source)
            np.save(output / "oof.npy", oof)
            np.save(output / "fold.npy", fold)
            (output / "metrics.json").write_text(
                json.dumps({"folds": scores, "pooled": pooled}, indent=2), encoding="utf-8"
            )

        print(store.publish(reservation, record, save), flush=True)


def write_notebook(trial: str, owner: str, output: Path, source: str) -> None:
    with_history = trial != "E00"
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


class Evaluation(NamedTuple):
    data_hash: str
    weeks: np.ndarray
    target: np.ndarray
    fold: np.ndarray
    source: bytes


def evaluation(data: Path) -> Evaluation:
    base = pl.read_parquet(shards(data, "train", "base"), columns=[WEEK, "target"])
    weeks, target = base[WEEK].to_numpy(), base["target"].to_numpy()
    fold = np.full(len(base), -1, dtype=np.int8)
    for index, (_, valid) in enumerate(fold_weeks(weeks, target)):
        fold[valid] = index
    return Evaluation(input_hash(data, "train"), weeks, target, fold, PIPELINE.read_bytes())


def verified_artifact(store: RunStore, record: ExperimentRecord, name: str) -> bytes:
    if name not in record.artifacts:
        raise ValueError(
            f"{record.run_id}: {name} の証跡がありません。新しい run ID で CV を再実行してください"
        )
    contents = (store.runs_dir / record.run_id / name).read_bytes()
    if hashlib.sha256(contents).hexdigest() != record.artifacts[name]:
        raise ValueError(f"{record.run_id}: {name} の SHA256 が一致しません")
    return contents


def verify_cv(store: RunStore, record: ExperimentRecord, trial: str, expected: Evaluation) -> None:
    if (
        record.competition_id != "005"
        or record.data_version != trial
        or record.metric != METRIC
        or record.greater_is_better is not True
        or record.split_version != SPLIT_VERSION
        or record.seed != PARAMS["seed"]
        or record.model != "LightGBM"
        or record.params != PARAMS
        or record.data_hash != expected.data_hash
        or len(record.fold_scores) != 3
    ):
        raise ValueError(f"{record.run_id}: CV run の評価条件が一致しません")
    fold = np.load(io.BytesIO(verified_artifact(store, record, "fold.npy")), allow_pickle=False)
    if (
        not np.array_equal(fold, expected.fold)
        or fold.dtype != expected.fold.dtype
        or array_sha256(fold) != record.split_hash
    ):
        raise ValueError(f"{record.run_id}: fold 割当が一致しません")
    pred = np.load(io.BytesIO(verified_artifact(store, record, "oof.npy")), allow_pickle=False)
    if pred.shape != fold.shape or pred.dtype.kind != "f":
        raise ValueError(f"{record.run_id}: OOF の形状または型が不正です")
    valid = fold >= 0
    validate_probabilities(np.column_stack((1 - pred[valid], pred[valid])), int(valid.sum()), 2)
    metrics = json.loads(verified_artifact(store, record, "metrics.json"))
    scores = []
    for index in range(3):
        rows = fold == index
        scores.append(stability(expected.weeks[rows], expected.target[rows], pred[rows])["score"])
    pooled = stability(expected.weeks[valid], expected.target[valid], pred[valid])["score"]
    if (
        len(metrics["folds"]) != 3
        or not np.allclose(scores, [m["score"] for m in metrics["folds"]], rtol=0, atol=1e-12)
        or not np.allclose(scores, record.fold_scores, rtol=0, atol=1e-12)
        or not np.isclose(np.mean(scores), record.oof_score, rtol=0, atol=1e-12)
        or not np.isclose(pooled, metrics["pooled"]["score"], rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{record.run_id}: OOF の再計算スコアが一致しません")
    if verified_artifact(store, record, "pipeline.py") != expected.source:
        raise ValueError(f"{record.run_id}: CV 時の pipeline.py と一致しません")


def export(
    store: RunStore,
    data: Path,
    trial: str,
    owner: str,
    run_id: str,
    selected: ExperimentRecord,
    expected: Evaluation,
) -> None:
    with store.reserve(run_id) as reservation:
        verify_cv(store, selected, trial, expected)
        source_run = selected.run_id
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
                **{
                    f"{source_run}/{name}": selected.artifacts[name]
                    for name in ("oof.npy", "fold.npy", "metrics.json", "pipeline.py")
                },
                "test_parquet": input_hash(data, "test"),
            },
            conclusion="oof_score は選定した CV run の参照値",
        )
        print(
            store.publish(
                reservation,
                record,
                lambda out: write_notebook(trial, owner, out, expected.source.decode("utf-8")),
            )
        )


def select_final(store: RunStore, expected: Evaluation, run_ids: list[str]) -> ExperimentRecord:
    records = [store.record(run_id) for run_id in run_ids]
    candidates = {record.data_version: record for record in records}
    if len(records) != len(TRIALS) or candidates.keys() != TRIALS.keys():
        raise ValueError("E00〜E04 の CV run を各1件指定してください")
    for trial, candidate in candidates.items():
        verify_cv(store, candidate, trial, expected)
    return max((candidates[trial] for trial in TRIALS), key=lambda c: c.oof_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=(*TRIALS, "best"))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--run-id")
    parser.add_argument("--owner", help="指定すると提出 Notebook を生成する")
    parser.add_argument("--source-run", help="全行再学習で参照する CV run")
    parser.add_argument("--candidate-run", action="append", help="best の CV 候補 ID。5件を指定")
    args = parser.parse_args()
    if args.trial == "best":
        if not args.owner or args.source_run:
            parser.error("best には --owner が必要です。--source-run は使えません")
        store = RunStore(ROOT)
        expected = evaluation(args.data)
        selected = select_final(store, expected, args.candidate_run or list(TRIALS))
        print(f"選定: {selected.run_id} {selected.oof_score:.8f}", flush=True)
        export(
            store,
            args.data,
            selected.data_version,
            args.owner,
            args.run_id or "F00",
            selected,
            expected,
        )
    elif args.candidate_run:
        parser.error("--candidate-run は best 専用です")
    elif args.owner:
        store = RunStore(ROOT)
        selected = store.record(args.source_run or args.trial)
        export(
            store,
            args.data,
            args.trial,
            args.owner,
            args.run_id or f"F-{args.trial}",
            selected,
            evaluation(args.data),
        )
    else:
        run(args.data, args.trial, args.run_id or args.trial)


if __name__ == "__main__":
    main()
