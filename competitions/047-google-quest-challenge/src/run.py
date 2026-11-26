"""Google QUEST の CV・選定・提出 Notebook 生成。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from groups import question_groups
from quest import PARAMS, correlations, predict, read_data, score, submission
from sklearn.model_selection import GroupKFold

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.kaggle_notebook import write_cpu_notebook
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
SLUG = "google-quest-challenge"
SPLIT = "quest-group-v1"
METHODS = {
    "B00": "結合した単語 TF-IDF と独立 Ridge",
    "E01": "フィールド別の単語 TF-IDF",
    "E02": "文字 TF-IDF の追加",
    "E03": "15成分のラベル PCA",
    "E04": "フィールド別 TF-IDF と経験分位点への写像",
}


def source(variant: str) -> SourceSpec:
    local = (ROOT / "src").relative_to(REPO).as_posix()
    shared = sorted((REPO / "src/kumite").rglob("*.py"))
    files = (
        "uv.lock",
        "pyproject.toml",
        f"{local}/run.py",
        f"{local}/quest.py",
        f"{local}/groups.py",
        *(path.relative_to(REPO).as_posix() for path in shared),
    )
    return SourceSpec(REPO, files, {"variant": variant, **PARAMS}, ("src", local))


def notebook_source(variant: str) -> str:
    library = Path(__file__).with_name("quest.py").read_text(encoding="utf-8")
    return (
        library
        + f"""

directory, = (
    path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv")
    if path.parent.name == "{SLUG}"
)
train, test, sample = read_data(directory)
targets = sample.columns[1:]
pred, _ = predict(train, test, train[targets].to_numpy(), {variant!r})
submission(sample, test, pred).to_csv("/kaggle/working/submission.csv", index=False)
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "select", "export", "eda"])
    parser.add_argument("--plan", default="initial")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--owner", default="shion5")
    parser.add_argument("--export-id", default="F00")
    args = parser.parse_args()
    data = ROOT / "data"
    train, test, sample = read_data(data)
    columns = sample.columns[1:]
    y = train[columns].to_numpy(dtype=float)
    assert y.shape == (len(train), 30) and np.isfinite(y).all()
    assert ((y >= 0) & (y <= 1)).all()
    groups = question_groups(train)
    splits = tuple(GroupKFold(n_splits=5).split(train, groups=groups))
    roles = split_roles(len(y), splits)
    if args.action == "eda":
        counts = np.unique(groups, return_counts=True)[1]
        print(
            "train/test/groups/duplicate_groups:",
            len(train),
            len(test),
            len(counts),
            sum(counts > 1),
        )
        print("site counts:", train["host"].value_counts().head().to_dict())
        print("fold sizes:", [len(valid) for _, valid in splits])
        print("fold means:", [round(float(y[valid].mean()), 4) for _, valid in splits])
        print("unique values per target:", train[columns].nunique().tolist())
        corr = train[columns].corr(method="spearman").to_numpy()
        print(
            "median absolute target correlation:", np.median(np.abs(corr[np.triu_indices(30, 1)]))
        )
        return
    plan_data = json.loads((ROOT / f"plan-{args.plan}.json").read_text(encoding="utf-8"))
    plan = SelectionPlan(tuple(plan_data["candidates"]), plan_data["tolerance"])
    receipt = ROOT / f"selection-{args.plan}.json"
    digest = array_sha256(
        np.asarray(
            [
                file_sha256(data / name)
                for name in ("train.csv", "test.csv", "sample_submission.csv")
            ]
        )
    )
    evaluation = Evaluation(
        "047", "mean Spearman", True, score, digest, SPLIT, array_sha256(roles), 42
    )
    store = RunStore(ROOT)

    def variant_for(run_id: str) -> str:
        return run_id.split("-", 1)[0]

    if args.action == "train":
        runs = args.runs or plan.candidates
        if not set(runs) <= set(plan.candidates):
            parser.error("候補計画にない run ID です")
        for run_id in runs:
            variant = variant_for(run_id)
            spec = source(variant)
            trial = Trial(
                METHODS[variant], METHODS[variant], "Ridge", "quest-tfidf-v1", dict(spec.settings)
            )

            def fit_fold(left: np.ndarray, right: np.ndarray, variant: str = variant) -> FoldOutput:
                pred, clip_rate = predict(train.iloc[left], train.iloc[right], y[left], variant)
                return FoldOutput(pred, np.empty(0), {"clip_rate": clip_rate})

            with reserve_cv(store, run_id, spec, {}, plan) as (reservation, start):
                before = perf_counter()
                result = cross_fit_splits(len(y), splits, fit_fold)

                def write_diagnostics(path: Path, oof: np.ndarray = result.oof) -> None:
                    metrics = dict(zip(columns, correlations(y, oof).tolist(), strict=True))
                    (path / "target-scores.json").write_text(
                        json.dumps(metrics, indent=2), encoding="utf-8"
                    )

                publish_cv_run(
                    store,
                    reservation,
                    evaluation,
                    trial,
                    y,
                    splits,
                    result,
                    perf_counter() - before,
                    write_diagnostics,
                    save_query=False,
                    gate=(spec, start, y.shape),
                )
            print(run_id, score(y, result.oof), flush=True)
        return

    assess = dense_oof(y, splits, score, expected_shape=y.shape, data_hash=digest)
    if args.action == "select":
        selection = select_runs(store, plan, assess)
        write_selection(receipt, selection)
        print(selection)
        return
    winner = json.loads(receipt.read_text(encoding="utf-8"))["winner"]
    variant = variant_for(winner)
    code = notebook_source(variant)

    def write(path: Path, _record: object) -> None:
        write_cpu_notebook(
            path,
            code,
            kernel_id=f"{args.owner}/kumite-quest-final",
            title="Kumite QUEST final",
            competition=SLUG,
        )
        pred, _ = predict(train, test, y, variant)
        submission(sample, test, pred).to_csv(path / "submission.csv", index=False)

    def validate(path: Path) -> None:
        nb = json.loads((path / "submit.ipynb").read_text(encoding="utf-8"))
        text = "".join(nb["cells"][0]["source"])
        if len(nb["cells"]) != 1 or text != code:
            raise ValueError("Invalid notebook source")
        compile(text, "submit.ipynb", "exec")
        meta = json.loads((path / "kernel-metadata.json").read_text(encoding="utf-8"))
        if meta["competition_sources"] != [SLUG] or meta["enable_internet"] or meta["enable_gpu"]:
            raise ValueError("Invalid notebook metadata")
        import pandas as pd

        actual = pd.read_csv(path / "submission.csv")
        if not actual.columns.equals(sample.columns) or not actual["qa_id"].equals(sample["qa_id"]):
            raise ValueError("Invalid submission IDs or column order")
        submission(sample, sample, actual.iloc[:, 1:].to_numpy())

    print(
        publish_export(
            store,
            args.export_id,
            plan=plan,
            receipt=receipt,
            assess=assess,
            source=source(variant),
            write=write,
            required_outputs=("submit.ipynb", "kernel-metadata.json", "submission.csv"),
            validate_outputs=validate,
        )
    )


if __name__ == "__main__":
    main()
