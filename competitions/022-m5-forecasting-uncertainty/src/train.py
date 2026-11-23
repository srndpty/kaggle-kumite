"""CV、選定、最終予測とその証跡を保存する。"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dummy import DATA, ROOT
from metric import evaluate, evaluation_inputs, score
from model import EXPERIMENTS, QUANTILES, forecast
from submission import write_submission

from kumite.experiments import ExperimentRecord
from kumite.m5 import (
    COMMON_FILES,
    EVALUATION_FILE,
    LEVELS,
    ORIGINS,
    PRIVATE_ORIGIN,
    PUBLIC_ORIGIN,
    VALIDATION_FILE,
    Hierarchy,
    Sales,
    load_sales,
)
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

STORE = RunStore(ROOT)
SPLIT = "m5-uncertainty-28d-v1"
WINDOWS = np.array(ORIGINS, dtype=np.int32)
SELECTION = ROOT / "selection.json"
SAMPLE = DATA / "sample_submission.csv"


def input_hash(*files: str) -> str:
    return files_sha256(DATA / name for name in (*COMMON_FILES, *files))


def hierarchy_for(data: Sales) -> Hierarchy:
    hierarchy = Hierarchy.from_series(data.series)
    if len(hierarchy.levels) != 42_840:
        raise ValueError("公式階層の系列数が一致しません")
    return hierarchy


def run_cv(experiment: str, run_id: str) -> None:
    with STORE.reserve(run_id) as reservation:
        started = time.perf_counter()
        version = code_version(ROOT.parents[1])
        data = load_sales(DATA)
        hierarchy = hierarchy_for(data)
        scores = []
        predictions = []
        for origin in ORIGINS:
            prediction = forecast(data, hierarchy, origin, experiment)
            scores.append(evaluate(data, hierarchy, origin, prediction))
            predictions.append(prediction)
            print(f"{experiment} d_{origin}: {scores[-1].mean():.8f}", flush=True)
        losses = np.stack(scores)
        hypothesis, change = EXPERIMENTS[experiment]
        record = ExperimentRecord(
            competition_id="022",
            run_id=run_id,
            metric="WSPL",
            greater_is_better=False,
            data_version=experiment,
            data_hash=input_hash(VALIDATION_FILE),
            split_version=SPLIT,
            split_hash=array_sha256(WINDOWS),
            seed=0,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model="seasonal_residual_quantiles",
            params={"quantiles": QUANTILES.tolist(), "experiment": experiment},
            fold_scores=losses.mean(axis=(1, 2)).tolist(),
            oof_score=float(losses.mean()),
            runtime_sec=time.perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
        )

        def save(path: Path) -> None:
            np.save(path / "origins.npy", WINDOWS)
            np.save(path / "scores.npy", losses)
            for origin, prediction in zip(ORIGINS, predictions, strict=True):
                np.save(path / f"d_{origin}.npy", prediction)
            hierarchy.labels.to_csv(path / "series.csv", index=False)

        print(STORE.publish(reservation, record, save))


def select(run_ids: list[str]) -> dict:
    records = [STORE.record(run_id) for run_id in run_ids]
    if (
        len(records) != len(EXPERIMENTS)
        or {record.data_version for record in records} != EXPERIMENTS.keys()
    ):
        raise ValueError("各実験の CV run を1件ずつ指定してください")
    expected = (input_hash(VALIDATION_FILE), SPLIT, array_sha256(WINDOWS))
    data = load_sales(DATA)
    hierarchy = hierarchy_for(data)
    inputs = {origin: evaluation_inputs(data, hierarchy, origin) for origin in ORIGINS}
    evidence = {}
    for record in records:
        if (record.data_hash, record.split_version, record.split_hash) != expected:
            raise ValueError(f"{record.run_id} の入力または分割が一致しません")
        losses = STORE.load_array(record.run_id, "scores.npy")
        if (
            losses.shape != (len(WINDOWS), len(LEVELS), len(QUANTILES))
            or not np.array_equal(STORE.load_array(record.run_id, "origins.npy"), WINDOWS)
            or not np.isfinite(losses).all()
            or not np.array_equal(losses.mean(axis=(1, 2)), record.fold_scores)
            or float(losses.mean()) != record.oof_score
        ):
            raise ValueError(f"{record.run_id} の WSPL が一致しません")
        for index, origin in enumerate(ORIGINS):
            name = f"d_{origin}.npy"
            prediction = STORE.load_array(record.run_id, name)
            actual, weight, scale = inputs[origin]
            replayed = score(actual, prediction, hierarchy, weight, scale)
            if not np.allclose(replayed, losses[index], rtol=1e-10, atol=1e-12):
                raise ValueError(f"{record.run_id} d_{origin} の再採点 WSPL が一致しません")
        evidence[record.run_id] = STORE.input_hashes(
            record.run_id, "scores.npy", *(f"d_{origin}.npy" for origin in ORIGINS)
        )
    best = min(records, key=lambda record: (record.oof_score, record.data_version))
    return {
        "selected": best.run_id,
        "experiment": best.data_version,
        "data_hash": best.data_hash,
        "split_hash": best.split_hash,
        "comparisons": {record.run_id: record.oof_score for record in records},
        "artifacts": evidence,
    }


def prepare_submission(
    experiment: str, run_id: str, cv_run: str, evidence: dict[str, str] | None = None
) -> None:
    with STORE.reserve(run_id) as reservation:
        started = time.perf_counter()
        version = code_version(ROOT.parents[1])
        source = STORE.record(cv_run)
        if (
            source.data_version != experiment
            or source.data_hash != input_hash(VALIDATION_FILE)
            or source.split_version != SPLIT
            or source.split_hash != array_sha256(WINDOWS)
        ):
            raise ValueError("生成元の CV run が一致しません")
        public_data = load_sales(DATA)
        public_hierarchy = hierarchy_for(public_data)
        public = forecast(public_data, public_hierarchy, PUBLIC_ORIGIN, experiment)
        del public_data
        private_data = load_sales(DATA, evaluation=True)
        private_hierarchy = hierarchy_for(private_data)
        private = forecast(private_data, private_hierarchy, PRIVATE_ORIGIN, experiment)
        # 予測の順序と重みは同じ階層キーに対応することを確かめる。
        if not public_hierarchy.labels.equals(private_hierarchy.labels):
            raise ValueError("公開・非公開データの階層キーが一致しません")
        public_check = evaluate(private_data, private_hierarchy, PUBLIC_ORIGIN, public)
        record = ExperimentRecord(
            competition_id="022",
            run_id=run_id,
            metric="WSPL",
            greater_is_better=False,
            data_version=experiment,
            data_hash=input_hash(VALIDATION_FILE, EVALUATION_FILE, SAMPLE.name),
            split_version=SPLIT,
            split_hash=source.split_hash,
            seed=0,
            hypothesis="固定した手法を外部評価する",
            changed_one_thing="提出用の起点で予測",
            model=source.model,
            params={**source.params, "source_run": cv_run},
            fold_scores=source.fold_scores,
            oof_score=source.oof_score,
            runtime_sec=time.perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
            inputs=evidence if evidence is not None else STORE.input_hashes(cv_run, "scores.npy"),
        )

        def save(path: Path) -> None:
            write_submission(
                path / "submission.csv",
                SAMPLE,
                (public_hierarchy, public),
                (private_hierarchy, private),
            )
            np.save(path / "public_scores.npy", public_check)

        print(STORE.publish(reservation, record, save))
        print(f"公開窓のローカル WSPL: {public_check.mean():.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv")
    cv.add_argument("experiments", nargs="+", choices=tuple(EXPERIMENTS))
    cv.add_argument("--run-id")
    selection = commands.add_parser("select")
    selection.add_argument("run_ids", nargs="+")
    selection.add_argument("--output", type=Path, default=SELECTION)
    selection.add_argument(
        "--audit-existing", type=Path, help="提出済みの選定記録と再採点結果を照合"
    )
    baseline = commands.add_parser("baseline")
    baseline.add_argument("cv_run")
    baseline.add_argument("--run-id", default="baseline-v1")
    final = commands.add_parser("final")
    final.add_argument("run_id")
    final.add_argument("--selection", type=Path, default=SELECTION)
    options = parser.parse_args()
    if options.command == "cv":
        if options.run_id and len(options.experiments) != 1:
            parser.error("--run-id は実験を1件指定した場合に使えます")
        for experiment in options.experiments:
            run_cv(experiment, options.run_id or experiment)
    elif options.command == "select":
        result = select(options.run_ids)
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        if options.audit_existing:
            original_text = options.audit_existing.read_text(encoding="utf-8")
            original = json.loads(original_text)
            historical_fields = {key: value for key, value in result.items() if key != "artifacts"}
            if original != historical_fields:
                raise ValueError("既存の選定記録と再採点結果が一致しません")
            result["audit"] = {
                "verified_at": timestamp,
                "original_selection_sha256": hashlib.sha256(
                    original_text.encode("utf-8")
                ).hexdigest(),
            }
        else:
            result["selected_at"] = timestamp
        with options.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
    elif options.command == "baseline":
        prepare_submission("E01", options.run_id, options.cv_run)
    else:
        result = json.loads(options.selection.read_text(encoding="utf-8"))
        verified = select(list(result["comparisons"]))
        comparable = {
            key: value for key, value in result.items() if key not in {"audit", "selected_at"}
        }
        if comparable != verified:
            raise ValueError("選定ファイルが CV の結果と一致しません")
        selected = verified["selected"]
        prepare_submission(
            verified["experiment"], options.run_id, selected, verified["artifacts"][selected]
        )


if __name__ == "__main__":
    main()
