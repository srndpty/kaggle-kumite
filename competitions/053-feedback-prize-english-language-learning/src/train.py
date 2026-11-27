"""固定分割で学習し、OOF を再採点できる形で保存する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import TARGETS, duplicate_key, fit_model, read_essays, validate_labels
from sklearn.model_selection import StratifiedGroupKFold

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, file_sha256
from kumite.quality_gates import SelectionPlan, SourceSpec, reserve_cv
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[1]
SEED = 42
SHARED = (
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
TRIALS = {
    "E00": ("base", "語彙分布から６軸の得点を推定できる"),
    "E01": ("bigram", "語の組み合わせが phraseology と syntax を改善する"),
    "E02": ("char", "語形や句読点が conventions と grammar を改善する"),
    "E03": ("length", "文章量と文構造が cohesion を改善する"),
    "E04": ("rbf", "語彙の非線形な類似度で MCRMSE が改善する"),
}


def mcrmse(labels: np.ndarray, predicted: np.ndarray) -> float:
    if (
        predicted.shape != labels.shape
        or labels.ndim != 2
        or labels.shape[1] != len(TARGETS)
        or len(labels) == 0
    ):
        raise ValueError("空でない６列の予測と正解が必要です")
    for values in (labels, predicted):
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("採点には有限な実数が必要です")
    errors = labels.astype(np.float64, copy=False) - predicted.astype(np.float64, copy=False)
    score = float(np.sqrt(np.mean(np.square(errors), axis=0)).mean())
    if not np.isfinite(score):
        raise ValueError("MCRMSE が非有限値になりました")
    return score


def load_cv():
    path = COMP / "data" / "train.csv"
    train = read_essays(path)
    texts = train["full_text"].fillna("").tolist()
    labels = train[list(TARGETS)].to_numpy(dtype=float)
    validate_labels(labels, len(texts))
    groups = np.asarray([duplicate_key(text) for text in texts])
    bins = pd.qcut(labels.mean(axis=1), 5, labels=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.arange(len(labels)), bins, groups))
    roles = split_roles(len(labels), splits)
    if not ((roles == 2).sum(axis=0) == 1).all():
        raise ValueError("全行を１回ずつ検証する必要があります")
    evaluation = Evaluation(
        "053", "MCRMSE", False, mcrmse, file_sha256(path), "v1", array_sha256(roles), SEED
    )
    return texts, labels, groups, splits, evaluation


def source_spec(block: str, plan_file: Path) -> SourceSpec:
    files = (Path(__file__), COMP / "src" / "model.py", COMP / "quality-gates.toml", plan_file)
    return SourceSpec(
        REPO_ROOT,
        (
            *(path.relative_to(REPO_ROOT).as_posix() for path in files),
            *(f"src/kumite/{name}.py" for name in SHARED),
            "pyproject.toml",
            "uv.lock",
        ),
        {"block": block, "seed": SEED, "split_version": "v1"},
        import_roots=("src", (COMP / "src").relative_to(REPO_ROOT).as_posix()),
    )


def read_plan(path: Path) -> SelectionPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SelectionPlan(tuple(raw["candidates"]), float(raw["tolerance"]))


def run(trial: str, run_id: str, plan_file: Path) -> None:
    texts, labels, _, splits, evaluation = load_cv()
    block, hypothesis = TRIALS[trial]
    source = source_spec(block, plan_file)
    store = RunStore(COMP)

    def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        start = perf_counter()
        predict = fit_model([texts[i] for i in train], labels[train], block)
        fit_sec = perf_counter() - start
        start = perf_counter()
        predictions = predict([texts[i] for i in valid])
        return FoldOutput(
            predictions,
            np.empty((0, len(TARGETS))),
            {"fit_sec": fit_sec, "predict_sec": perf_counter() - start},
        )

    def write_metrics(output: Path) -> None:
        rmse = np.sqrt(np.mean(np.square(labels - result.oof), axis=0))
        metrics = dict(zip(TARGETS, rmse.tolist(), strict=True))
        (output / "target_rmse.json").write_text(json.dumps(metrics), encoding="utf-8")

    with reserve_cv(store, run_id, source, {}, read_plan(plan_file)) as (reservation, evidence):
        start = perf_counter()
        result = cross_fit_splits(len(labels), splits, fit_fold)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            Trial(
                hypothesis,
                block,
                "TF-IDF RBF KernelRidge" if block == "rbf" else "TF-IDF Ridge",
                "official-v1",
                {"block": block},
            ),
            labels,
            splits,
            result,
            perf_counter() - start,
            write_metrics,
            save_query=False,
            gate=(source, evidence, labels.shape),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--plan", type=Path, default=COMP / "selection-plan.json")
    args = parser.parse_args()
    run(args.trial, args.run_id or args.trial, args.plan.resolve())


if __name__ == "__main__":
    main()
