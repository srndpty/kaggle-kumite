"""固定被験者分割で学習し、OOF とテスト予測を保存する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from pipeline import (
    EVENTS,
    RATES,
    annotated,
    class_ap,
    features,
    make_submission,
    model_groups,
    predict,
    predict_recordings,
    read_recording,
    recordings,
    score,
)
from prepare import ROOT, SEED

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256, files_sha256, peak_memory_mb
from kumite.runs import RunStore

TRIALS = {
    "E00": ("クラス別陽性率を基準とする", "学習 fold のクラス別陽性率"),
    "E01": ("瞬時値から歩行凍結を識別できる", "3軸の瞬時値と収録種別を使う LightGBM"),
    "E02": ("1秒の文脈で AP が上がる", "過去1秒の軸別平均・標準偏差を追加"),
    "E03": ("前後の文脈で StartHesitation AP が上がる", "1秒窓の位置を中央揃えに変更"),
    "E04": ("長い窓で Turn と Walking の AP が上がる", "過去窓を1秒から3秒へ延長"),
    "E05": ("収録条件ごとの学習で AP が上がる", "E04 を収録種別ごとのモデルに変更"),
}
THREADS = 8
ROUNDS = 100
STRIDE = 8


def fit(
    paths: list[Path], split: pd.DataFrame, trial: str, *, by_kind: bool = False
) -> tuple[dict[str, list[float | str]], np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    cohorts = split["fold"].unique().tolist()
    train_pos = np.zeros((len(cohorts), len(EVENTS)), dtype=np.int64)
    train_count = np.zeros(len(cohorts), dtype=np.int64)
    for path, fold in zip(paths, split["fold"].to_numpy(dtype=int), strict=True):
        kind = path.parent.name
        frame = read_recording(path, labeled=True)
        valid = annotated(frame, kind)
        labels = frame[EVENTS].to_numpy()
        train_pos[fold] += labels[valid].sum(axis=0, dtype=np.int64)
        train_count[fold] += valid.sum()
        if trial != "E00":
            sampled = valid & (frame["Time"].to_numpy() % STRIDE == 0)
            x_parts.append(features(frame, kind, trial)[sampled])
            y_parts.append(labels[sampled])
            fold_parts.append(np.full(int(sampled.sum()), fold, dtype=np.int8))
            kind_parts.append(np.full(int(sampled.sum()), kind == "tdcsfog"))

    models: dict[str, list[float | str]] = {}
    if trial == "E00":
        for fold in cohorts:
            models[str(fold)] = (
                (train_pos.sum(axis=0) - train_pos[fold]) / (train_count.sum() - train_count[fold])
            ).tolist()
        return models, train_count

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    fold_ids = np.concatenate(fold_parts)
    tdcs_rows = np.concatenate(kind_parts)
    for fold in sorted(cohorts):
        for kind in tuple(RATES) if by_kind else ("",):
            training = fold_ids != fold
            if kind:
                training &= tdcs_rows == (kind == "tdcsfog")
            key = f"{fold}:{kind}" if kind else str(fold)
            models[key] = fit_group(x[training], y[training], key)
        print(f"trained fold {fold}", flush=True)
    return models, train_count


def fit_group(x: np.ndarray, labels: np.ndarray, key: str) -> list[float | str]:
    models: list[float | str] = []
    for label in labels.T:
        if len(np.unique(label)) != 2:
            raise ValueError(f"Training group {key} lacks a class")
        model = lgb.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "seed": SEED,
                "num_threads": THREADS,
                "deterministic": True,
                "force_col_wise": True,
                "learning_rate": 0.08,
                "num_leaves": 31,
            },
            lgb.Dataset(x, label=label),
            num_boost_round=ROUNDS,
        )
        models.append(model.model_to_string())
    return models


def validate_predictions(predictions: np.ndarray) -> None:
    if not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise ValueError("Invalid event probabilities")


def out_of_fold(
    paths: list[Path],
    split: pd.DataFrame,
    trial: str,
    models: dict[str, list[float | str]],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictors = model_groups(models)
    labels = np.empty((row_count, len(EVENTS)), dtype=np.uint8)
    predictions = np.empty_like(labels, dtype=np.float32)
    start = 0
    for path, fold in zip(paths, split["fold"].to_numpy(dtype=int), strict=True):
        kind = path.parent.name
        frame = read_recording(path, labeled=True)
        mask = annotated(frame, kind)
        stop = start + int(mask.sum())
        if stop - start != split.at[path.stem, "annotated"]:
            raise ValueError(f"Annotation count changed for {path.stem}")
        labels[start:stop] = frame.loc[mask, EVENTS].to_numpy()
        predictions[start:stop] = predict(
            predictors[kind][str(fold)], features(frame, kind, trial)[mask]
        )
        start = stop
    if start != row_count:
        raise ValueError("OOF row count mismatch")
    validate_predictions(predictions)
    return labels, predictions


def evaluate(
    split: pd.DataFrame, labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, list[float], dict[str, list[float | None]]]:
    rows = split["annotated"].to_numpy()
    folds = np.repeat(split["fold"].to_numpy(dtype=np.int8), rows)
    kinds = np.repeat(split["kind"].eq("tdcsfog").to_numpy(), rows)
    fold_scores = [
        score(labels[folds == fold], predictions[folds == fold])
        for fold in sorted(split["fold"].unique())
    ]
    by_class = {"all": class_ap(labels, predictions)}
    by_class["tdcsfog"] = class_ap(labels[kinds], predictions[kinds])
    by_class["defog"] = class_ap(labels[~kinds], predictions[~kinds])
    values = by_class["all"]
    if any(value is None for value in values):
        raise ValueError("Missing class in combined OOF")
    return float(np.mean([value for value in values if value is not None])), fold_scores, by_class


def run(run_id: str, trial: str, data: Path, split_path: Path) -> None:
    started = perf_counter()
    paths = recordings(data, "train")
    test_paths = recordings(data, "test")
    sample_path = data / "sample_submission.csv"
    split = pd.read_csv(split_path, dtype={"Id": str, "subject": str}).set_index("Id")
    if not split.index.is_unique or set(split.index) != {path.stem for path in paths}:
        raise ValueError("Training recordings do not match the split")
    split = split.loc[[path.stem for path in paths]]
    if split.groupby("subject")["fold"].nunique().max() != 1:
        raise ValueError("Subject crosses folds")
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        data_hash = files_sha256(
            [
                *paths,
                *test_paths,
                sample_path,
                data / "defog_metadata.csv",
                data / "tdcsfog_metadata.csv",
            ]
        )
        split_hash = file_sha256(split_path)
        by_kind = trial == "E05"
        feature_trial = "E04" if by_kind else trial
        if by_kind:
            baseline = store.record(feature_trial)
            if (baseline.data_hash, baseline.split_hash) != (data_hash, split_hash):
                raise ValueError("E05 and E04 must use the same inputs and split")
        models, counts = fit(paths, split, feature_trial, by_kind=by_kind)
        labels, predictions = out_of_fold(paths, split, feature_trial, models, int(counts.sum()))
        ids, test_predictions = predict_recordings(test_paths, feature_trial, models)
        sample = pd.read_csv(sample_path, dtype={"Id": str})
        submission = make_submission(sample, ids, test_predictions)
        oof, folds, detail = evaluate(split, labels, predictions)
        hypothesis, change = TRIALS[trial]
        record = ExperimentRecord(
            competition_id=ROOT.name,
            run_id=run_id,
            metric="macro_average_precision",
            greater_is_better=True,
            data_version="official-labeled-recordings",
            data_hash=data_hash,
            split_version=f"subject-{len(counts)}fold-v1",
            split_hash=split_hash,
            seed=SEED,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model="constant" if trial == "E00" else "LightGBM",
            params={
                "trial": trial,
                "feature_trial": feature_trial,
                "stride": STRIDE,
                "rounds": ROUNDS,
                "threads": THREADS,
            },
            inputs=store.input_hashes(feature_trial, "split.csv") if by_kind else {},
            fold_scores=folds,
            oof_score=oof,
            runtime_sec=perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=code_version(REPO_ROOT),
        )

        def write(out: Path) -> None:
            split.to_csv(out / "split.csv")
            np.save(out / "labels.npy", labels)
            np.save(out / "oof.npy", predictions)
            submission.to_csv(out / "submission.csv", index=False)
            (out / "models.json").write_text(json.dumps(models), encoding="utf-8")
            (out / "ap.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

        store.publish(reservation, record, write)
        print(f"{run_id} OOF {oof:.6f} {detail}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/split.csv")
    args = parser.parse_args()
    trial = args.trial or args.run_id
    if trial not in TRIALS:
        parser.error("--trial is required for custom run IDs")
    run(args.run_id, trial, args.data, args.split)


if __name__ == "__main__":
    main()
