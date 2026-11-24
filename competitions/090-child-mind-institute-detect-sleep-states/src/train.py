"""固定分割で学習し、イベント予測と再実行用の成果物を保存する。"""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import polars as pl
from metric import score
from pipeline import EVENTS, bins, candidates, limit_candidates, matrix, submission
from sklearn.model_selection import KFold

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256, files_sha256, peak_memory_mb
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPLIT = "series-3fold-v1"
TRIALS = {
    "E00": ("基準", False, 0, 15),
    "E01": ("前後30分の特徴量", True, 0, 15),
    "E02": ("正例の前後3分拡張", False, 3, 15),
    "E03": ("重複抑制を60分に拡張", False, 0, 60),
    "E04": ("現地時刻の周期特徴量", False, 0, 15),
}


def split(frame: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    ids = frame["series_id"].unique().sort()
    folds = np.empty(len(ids), dtype=np.int8)
    for fold, (_, valid) in enumerate(KFold(3, shuffle=True, random_state=seed).split(ids)):
        folds[valid] = fold
    return pl.DataFrame({"series_id": ids, "fold": folds})


def labels(frame: pl.DataFrame, truth: pl.DataFrame, event: str, radius: int) -> np.ndarray:
    result = []
    annotations = truth.partition_by("series_id", as_dict=True)
    for key, rows in frame.partition_by("series_id", as_dict=True, maintain_order=True).items():
        y = np.full(len(rows), -1, dtype=np.int8)
        actual = annotations.get(key)
        if actual is not None:
            steps = rows["minute"].to_numpy()
            for night in actual.partition_by("night"):
                observed = night.drop_nulls("step")
                if observed.height == 2:
                    onset = int(observed.filter(pl.col("event") == "onset")["step"].item())
                    wakeup = int(observed.filter(pl.col("event") == "wakeup")["step"].item())
                    if onset < wakeup:
                        y[(steps >= onset // 12 - 180) & (steps <= wakeup // 12 + 180)] = 0
            for step in actual.filter(pl.col("event") == event)["step"].drop_nulls().to_list():
                y[np.abs(steps - step // 12) <= radius] = 1
        result.append(y)
    return np.concatenate(result)


def fit(
    frame: pl.DataFrame,
    truth: pl.DataFrame,
    split_ids: pl.DataFrame,
    out: Path,
    long_window: bool,
    radius: int,
    clock: bool,
    seed: int,
) -> pl.DataFrame:
    x = matrix(frame, long_window, clock)
    row_folds = frame.join(split_ids, on="series_id", maintain_order="left")["fold"].to_numpy()
    predictions = {}
    for event in EVENTS:
        y = labels(frame, truth, event, radius)
        oof = np.full(len(frame), np.nan)
        eligible = (y == 1) | ((y == 0) & (np.arange(len(y)) % 4 == 0))
        for fold in range(3):
            training = eligible & (row_folds != fold)
            valid = row_folds == fold
            model = lgb.train(
                {
                    "objective": "binary",
                    "metric": "None",
                    "verbosity": -1,
                    "seed": seed,
                    "num_threads": 8,
                    "deterministic": True,
                    "force_col_wise": True,
                    "learning_rate": 0.08,
                    "num_leaves": 31,
                },
                lgb.Dataset(x[training], label=y[training]),
                num_boost_round=120,
            )
            model.save_model(str(out / f"{event}-{fold}.txt"))
            oof[valid] = model.predict(x[valid], num_threads=8)
        if not (np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()):
            raise ValueError("Invalid OOF probabilities")
        predictions[event] = oof
    return frame.select("series_id", "segment", "step").with_columns(
        *(pl.Series(event, prediction) for event, prediction in predictions.items())
    )


def run(run_id: str, trial: str, seed: int, capped: bool) -> None:
    started = perf_counter()
    description, long_window, radius, suppression = TRIALS[trial]
    clock = trial == "E04"
    if trial == "E03" and seed != SEED:
        raise ValueError("E03 reuses the fixed E00 split")
    if capped and (trial not in {"E00", "E04"} or seed != SEED):
        raise ValueError("Capped comparisons reuse the main E00 or E04 run")
    store = RunStore(ROOT)
    train_file = ROOT / "data/train_series.parquet"
    event_file = ROOT / "data/train_events.csv"
    truth = pl.read_csv(event_file, schema_overrides={"series_id": pl.String})
    data_hash = files_sha256((train_file, event_file))
    inputs: dict[str, str] = {}
    with store.reserve(run_id) as reservation:
        out = ROOT / "outputs" / f".training-{uuid4().hex}"
        out.mkdir(parents=True)
        try:
            if trial == "E03" or capped:
                source_id = trial if capped else "E00"
                baseline = store.record(source_id)
                if baseline.data_hash != data_hash:
                    raise ValueError("Baseline data mismatch")
                for name in (
                    "bins.parquet",
                    "split.csv",
                    *(f"{e}-{f}.txt" for e in EVENTS for f in range(3)),
                ):
                    shutil.copyfile(store.artifact_path(source_id, name), out / name)
                inputs = store.input_hashes(source_id, "bins.parquet", "split.csv")
                probabilities = pl.read_parquet(out / "bins.parquet")
                split_ids = pl.read_csv(out / "split.csv")
            else:
                frame = bins(train_file, clock)
                split_ids = split(frame, seed)
                split_ids.write_csv(out / "split.csv")
                probabilities = fit(frame, truth, split_ids, out, long_window, radius, clock, seed)
                probabilities.write_parquet(out / "bins.parquet")
            detections = candidates(probabilities, suppression)
            if capped:
                detections = limit_candidates(detections, probabilities)
                description = "各系列・各イベントの日数あたり候補上限を8件にする"
            submission(detections, train_file).write_csv(out / "oof.csv")
            detail = score(truth, detections)
            fold_scores = []
            for fold in range(3):
                ids = split_ids.filter(pl.col("fold") == fold).select("series_id")
                fold_detail = score(
                    truth.join(ids, on="series_id"), detections.join(ids, on="series_id")
                )
                fold_scores.append(float(np.mean(list(fold_detail.values()))))
            (out / "ap.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
            record = ExperimentRecord(
                competition_id=ROOT.name,
                run_id=run_id,
                metric="event_detection_ap",
                greater_is_better=True,
                data_version="official-1min",
                data_hash=data_hash,
                split_version=SPLIT if seed == SEED else f"{SPLIT}-seed{seed}",
                split_hash=file_sha256(out / "split.csv"),
                seed=seed,
                hypothesis=description,
                changed_one_thing=description,
                model="LightGBM",
                params={
                    "long_window": long_window,
                    "clock": clock,
                    "capped": capped,
                    "radius": radius,
                    "suppression": suppression,
                    "boost_rounds": 120,
                    "negative_stride": 4,
                    "night_margin_minutes": 180,
                },
                fold_scores=fold_scores,
                oof_score=float(np.mean(list(detail.values()))),
                runtime_sec=perf_counter() - started,
                peak_memory_mb=peak_memory_mb(),
                code_version=code_version(REPO_ROOT),
                inputs=inputs,
            )

            def write(destination: Path) -> None:
                for file in out.iterdir():
                    shutil.copyfile(file, destination / file.name)

            store.publish(reservation, record, write)
            print(f"{run_id}: {record.oof_score:.6f} {fold_scores}", flush=True)
        finally:
            shutil.rmtree(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capped", action="store_true")
    args = parser.parse_args()
    run(args.run_id or args.trial, args.trial, args.seed, args.capped)
