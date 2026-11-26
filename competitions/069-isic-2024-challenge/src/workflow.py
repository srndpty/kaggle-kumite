"""CV、選定、最終 export の実行入口。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import NEGATIVE_RATIO, ROUNDS, SEED, features, fit, predict, settings
from submission import validate_export, write_export
from validation import SPLIT_VERSION, make_splits, pauc

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, file_sha256
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    dense_oof,
    publish_export,
    reserve_cv,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
TRAIN = ROOT / "data/train-metadata.csv"
TEST = ROOT / "data/test-metadata.csv"
SAMPLE = ROOT / "data/sample_submission.csv"
TRIALS = {
    "B00": "病変単位の診察時メタデータ",
    "E01": "病変サイズと色差の派生特徴",
    "E02": "患者内の相対特徴",
    "E03": "陽性学習重みを5倍",
    "E04": "学習 fold の陰性を陽性の5倍まで抽出",
}


def source(variant: str, plan: Path) -> SourceSpec:
    prefix = ROOT.relative_to(REPO).as_posix()
    local = ("workflow", "model", "submission", "validation")
    shared = (
        "__init__",
        "crossfit",
        "cv_run",
        "experiments",
        "paths",
        "provenance",
        "quality_gates",
        "runs",
        "source_audit",
    )
    return SourceSpec(
        REPO,
        (
            *(f"{prefix}/src/{name}.py" for name in local),
            *(f"src/kumite/{name}.py" for name in shared),
            "pyproject.toml",
            "uv.lock",
            f"{prefix}/quality-gates.toml",
            plan.relative_to(REPO).as_posix(),
        ),
        {
            "variant": variant,
            "rounds": ROUNDS,
            "model": settings(variant),
            "negative_ratio": NEGATIVE_RATIO if variant == "E04" else None,
        },
        ("src", f"{prefix}/src"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "select", "export"))
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--plan", type=Path, default=ROOT / "selection-plan.json")
    parser.add_argument("--receipt", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    if args.action != "select" and args.run_id is None:
        parser.error("train と export には run_id が必要です")
    plan_path = args.plan.resolve()
    config = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plan = SelectionPlan(tuple(config["candidates"]), config["tolerance"])

    data_hash = file_sha256(TRAIN)
    frame = pd.read_csv(TRAIN, low_memory=False)
    if frame["isic_id"].isna().any() or not frame["isic_id"].is_unique:
        raise ValueError("学習病変 ID が欠落または重複しています")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("学習ラベルが二値ではありません")
    y = frame["target"].to_numpy(dtype=np.int8)
    splits = make_splits(frame, y)
    store = RunStore(ROOT)
    assess = dense_oof(y, splits, pauc, expected_shape=y.shape, data_hash=data_hash)

    if args.action == "select":
        selection = select_runs(store, plan, assess)
        write_selection(args.receipt, selection)
        print(selection)
        return

    if args.action == "export":
        variant = store.record(select_runs(store, plan, assess).winner).params["variant"]
        sample = pd.read_csv(SAMPLE)
        test = pd.read_csv(TEST)
        print(
            publish_export(
                store,
                args.run_id,
                plan=plan,
                receipt=args.receipt,
                assess=assess,
                source=source(variant, plan_path),
                write=lambda path, _: write_export(path, frame, y, test, sample, variant),
                required_outputs=("model.txt", "submit.ipynb", "sample-predictions.csv"),
                validate_outputs=lambda path: validate_export(path, sample),
            )
        )
        return

    run_id = args.run_id
    if run_id not in plan.candidates:
        raise ValueError("学習 run が事前計画に含まれていません")
    variant = run_id.split("-")[0]
    trial = Trial(
        TRIALS[variant],
        TRIALS[variant],
        "CPU LightGBM",
        "official-metadata-v1",
        {"variant": variant, "rounds": ROUNDS, **settings(variant)},
    )
    evaluation = Evaluation(
        "069",
        "pAUC_TPR80",
        True,
        pauc,
        data_hash,
        SPLIT_VERSION,
        array_sha256(split_roles(len(y), splits)),
        SEED,
    )
    spec = source(variant, plan_path)
    with reserve_cv(store, run_id, spec, {}, plan) as (reservation, start):
        started = perf_counter()
        x = features(frame, variant)

        def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
            model = fit(x.iloc[train], y[train], variant)
            return FoldOutput(
                predict(model, x.iloc[valid]),
                np.empty(0),
                {
                    "valid_positives": int(y[valid].sum()),
                    "valid_patients": int(frame["patient_id"].iloc[valid].nunique()),
                },
            )

        result = cross_fit_splits(len(y), splits, fit_fold)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            perf_counter() - started,
            lambda path: None,
            save_query=False,
            gate=(spec, start, y.shape),
        )


if __name__ == "__main__":
    main()
