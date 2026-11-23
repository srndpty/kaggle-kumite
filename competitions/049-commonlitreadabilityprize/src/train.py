"""固定分割で学習し、検証予測と選定の証跡を保存する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import FILES, HYPOTHESES, load, predict, structure, write_submission
from splits import SEED, VERSION, make_folds

from kumite.checks import validate_regression
from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FOLDS = ROOT / "outputs" / "folds.csv"


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> None:
    folds = make_folds(train)
    FOLDS.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS, index=False)
    print(f"train={len(train)} test={len(test)} groups={folds['group'].nunique()}")
    print(folds.groupby("fold").size().to_string())
    for column in ("url_legal", "license"):
        print(f"{column} 欠損: train={train[column].isna().sum()} test={test[column].isna().sum()}")
    print("目的値の分位:", train["target"].quantile([0, 0.1, 0.5, 0.9, 1]).tolist())
    print(
        "語数の分位:",
        np.quantile(structure(train["excerpt"].tolist())[:, 0], [0, 0.1, 0.5, 0.9, 1]),
    )


def run_cv(
    store: RunStore,
    train: pd.DataFrame,
    folds: pd.DataFrame,
    evaluation: Evaluation,
    trial: str,
    run_id: str,
) -> None:
    texts: list[str] = train["excerpt"].tolist()
    target = train["target"].to_numpy(dtype=float)
    splits = folds_to_splits(folds["fold"].to_numpy())
    coverage = np.full(len(train), np.nan)

    def fit(training: np.ndarray, validation: np.ndarray) -> FoldOutput:
        pred, fold_coverage = predict(
            [texts[i] for i in training], target[training], [texts[i] for i in validation], trial
        )
        coverage[validation] = fold_coverage
        return FoldOutput(pred, np.empty(0))

    def write(output: Path) -> None:
        folds.to_csv(output / "folds.csv", index=False)
        np.save(output / "coverage.npy", coverage)

    with store.reserve(run_id) as reservation:
        start = perf_counter()
        result = cross_fit_splits(len(train), splits, fit)
        validate_regression(result.oof, len(train))
        validate_regression(coverage, len(train))
        estimators = {
            "E00": ("mean", {}),
            "E05": (
                "LinearSVR",
                {"C": 1.0, "epsilon": 0.1, "dual": True, "max_iter": 10000, "random_state": 42},
            ),
        }
        estimator, params = estimators.get(trial, ("Ridge", {"alpha": 1.0, "solver": "lsqr"}))
        metadata = Trial(
            HYPOTHESES[trial],
            HYPOTHESES[trial],
            estimator,
            "kaggle-official",
            {"trial": trial, **params},
        )
        publish_cv_run(
            store,
            reservation,
            evaluation,
            metadata,
            target,
            splits,
            result,
            perf_counter() - start,
            write,
            save_query=False,
        )


def bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> list[float]:
    _, labels = np.unique(groups, return_inverse=True)
    count = np.bincount(labels)
    errors = [np.bincount(labels, weights=(y - pred) ** 2) for pred in (a, b)]
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(2000):
        sample = rng.integers(0, len(count), size=len(count))
        n_rows = count[sample].sum()
        values.append(
            np.sqrt(errors[0][sample].sum() / n_rows) - np.sqrt(errors[1][sample].sum() / n_rows)
        )
    return np.quantile(values, [0.025, 0.975]).tolist()


def select(
    store: RunStore,
    train: pd.DataFrame,
    folds: pd.DataFrame,
    evaluation: Evaluation,
    run_ids: list[str],
) -> dict[str, object]:
    y = train["target"].to_numpy(dtype=float)
    scores: list[tuple[float, str]] = []
    predictions: dict[str, np.ndarray] = {}
    for run_id in run_ids:
        record = store.record(run_id)
        if (record.data_hash, record.split_hash, record.split_version) != (
            evaluation.data_hash,
            evaluation.split_hash,
            evaluation.split_version,
        ):
            raise ValueError(f"{run_id}: データまたは検証分割が異なります")
        pred = store.load_array(run_id, "oof.npy")
        validate_regression(pred, len(y))
        scores.append((rmse(y, pred), run_id))
        predictions[run_id] = pred
    scores.sort()
    _, best = scores[0]
    _, runner_up = scores[1]
    coverage = store.load_array(best, "coverage.npy")
    slices: dict[str, list[dict[str, float | int]]] = {}
    for name, values in (
        ("words", structure(train["excerpt"].tolist())[:, 0]),
        ("target", y),
        ("coverage", coverage),
    ):
        bands = np.digitize(values, np.unique(np.quantile(values, [0.25, 0.5, 0.75])), right=True)
        slices[name] = [
            {"rows": int(mask.sum()), "rmse": rmse(y[mask], predictions[best][mask])}
            for band in np.unique(bands)
            if (mask := bands == band).any()
        ]
    selected = store.record(best)
    return {
        "run_id": best,
        "trial": selected.params["trial"],
        "oof_rmse": dict((run_id, score) for score, run_id in scores),
        "data_hash": evaluation.data_hash,
        "split_hash": evaluation.split_hash,
        "runner_up": runner_up,
        "delta_ci95": bootstrap(
            y, predictions[best], predictions[runner_up], folds["group"].to_numpy()
        ),
        "slices": slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    cv = commands.add_parser("cv")
    cv.add_argument("trial", choices=HYPOTHESES)
    cv.add_argument("--run-id")
    selection = commands.add_parser("select")
    selection.add_argument("runs", nargs="+")
    selection.add_argument("--output", type=Path, default=ROOT / "selection.json")
    final = commands.add_parser("final")
    final.add_argument("selection", type=Path)
    final.add_argument("output", type=Path)
    args = parser.parse_args()
    train, test, sample = load(DATA)
    if args.command == "prepare":
        prepare(train, test)
        return
    folds = pd.read_csv(FOLDS, dtype={"id": str})
    if not folds["id"].equals(train["id"]):
        raise ValueError("分割 CSV と学習 ID の行順が異なります")
    evaluation = Evaluation(
        "049",
        "RMSE",
        False,
        rmse,
        files_sha256(DATA / name for name in FILES),
        VERSION,
        files_sha256([FOLDS]),
        SEED,
    )
    store = RunStore(ROOT)
    if args.command == "cv":
        run_cv(store, train, folds, evaluation, args.trial, args.run_id or args.trial)
    elif args.command == "select":
        if len(set(args.runs)) < 2 or len(args.runs) != len(set(args.runs)):
            parser.error("異なる run を2件以上指定してください")
        selected = select(store, train, folds, evaluation, args.runs)
        args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(selected, indent=2))
    else:
        selected = json.loads(args.selection.read_text(encoding="utf-8"))
        if selected != select(store, train, folds, evaluation, list(selected["oof_rmse"])):
            raise ValueError("選定記録と保存済み OOF が一致しません")
        pred, _ = predict(
            train["excerpt"].tolist(),
            train["target"].to_numpy(),
            test["excerpt"].tolist(),
            selected["trial"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_submission(test, sample, pred, args.output)


if __name__ == "__main__":
    main()
