"""エッセイ単位の固定 CV、保存 OOF の再採点と選定。"""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import (
    CLASSES,
    ESSAY,
    ROLE,
    TEXT,
    check_probabilities,
    fit_model,
    labels_for,
    read_discourse,
)
from sklearn.model_selection import StratifiedGroupKFold

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, files_sha256
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    dense_oof,
    reserve_cv,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[1]
SEED = 42
TRIALS = {
    "E00": ("base", "論証本文の語彙から有効性を推定できる"),
    "E01": ("role", "論証種別を追加すると LogLoss が下がる"),
    "E02": ("char", "文字 n-gram が単語特徴を補う"),
    "E03": ("context", "他の論証文脈が短い論証の予測を改善する"),
    "E04": ("boundary", "論証の冒頭と末尾の語彙を区別すると LogLoss が下がる"),
}
RUN_TRIALS = {
    **{f"E{i:02}": f"E{i:02}" for i in range(4)},
    **{f"E{i:02}-final": f"E{i:02}" for i in range(5)},
    "E00-external": "E00",
}
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


def logloss_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    check_probabilities(scores, len(labels))
    if (
        labels.ndim != 1
        or not len(labels)
        or labels.dtype.kind not in "iu"
        or ((labels < 0) | (labels >= len(CLASSES))).any()
    ):
        raise ValueError("有効なクラス ID が必要です")
    return -np.log(np.clip(scores[np.arange(len(labels)), labels], 1e-15, 1 - 1e-15))


def logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(logloss_rows(labels, scores).mean())


def group_keys(frame: pd.DataFrame, directory: Path) -> tuple[np.ndarray, list[Path]]:
    paths = [directory / f"{key}.txt" for key in sorted(set(frame[ESSAY]))]
    by_essay = {}
    for path in paths:
        normalized = " ".join(
            unicodedata.normalize("NFKC", path.read_text(encoding="utf-8-sig")).casefold().split()
        )
        if not normalized:
            raise ValueError(f"エッセイ本文が空です: {path}")
        by_essay[path.stem] = hashlib.sha256(normalized.encode()).hexdigest()
    return frame[ESSAY].map(by_essay).to_numpy(), paths


def make_splits(labels: np.ndarray, groups: np.ndarray):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.arange(len(labels)), labels, groups))
    roles = split_roles(len(labels), splits)
    if not ((roles == 2).sum(axis=0) == 1).all():
        raise ValueError("各論証を１回ずつ検証する必要があります")
    return splits, roles


def load_cv():
    path = COMP / "data/train.csv"
    frame = read_discourse(path)
    labels = labels_for(frame)
    groups, essays = group_keys(frame, COMP / "data/train")
    splits, roles = make_splits(labels, groups)
    evaluation = Evaluation(
        "052",
        "LogLoss",
        False,
        logloss,
        files_sha256([path, *essays]),
        "v1",
        array_sha256(roles),
        SEED,
    )
    return frame, labels, groups, splits, evaluation


def source_spec(block: str, plan_file: Path) -> SourceSpec:
    files = (Path(__file__), COMP / "src/model.py", COMP / "quality-gates.toml", plan_file)
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
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return SelectionPlan(tuple(raw["candidates"]), float(raw["tolerance"]))


def assess_for(labels: np.ndarray, splits: list, evaluation: Evaluation):
    return dense_oof(
        labels,
        splits,
        logloss,
        expected_shape=(len(labels), len(CLASSES)),
        data_hash=evaluation.data_hash,
    )


def trial_for(run_id: str) -> str:
    if run_id not in RUN_TRIALS:
        raise ValueError(f"未登録の run ID: {run_id}")
    return RUN_TRIALS[run_id]


def check_candidates(store: RunStore, plan: SelectionPlan) -> None:
    for candidate in plan.candidates:
        if store.record(candidate).params["block"] != TRIALS[trial_for(candidate)][0]:
            raise ValueError(f"候補の構成が一致しません: {candidate}")


def run(trial: str, run_id: str, path: Path) -> None:
    plan = read_plan(path)
    if trial_for(run_id) != trial or run_id not in plan.candidates:
        raise ValueError("run ID が候補計画または trial と一致しません")
    frame, labels, _, splits, evaluation = load_cv()
    block, hypothesis = TRIALS[trial]
    source, store = source_spec(block, path), RunStore(COMP)

    def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        start = perf_counter()
        predict = fit_model(frame.iloc[train], block)
        fit_sec = perf_counter() - start
        start = perf_counter()
        scores = predict(frame.iloc[valid].drop(columns="discourse_effectiveness"))
        return FoldOutput(
            scores,
            np.empty((0, len(CLASSES))),
            {"fit_sec": fit_sec, "predict_sec": perf_counter() - start},
        )

    with reserve_cv(store, run_id, source, {}, plan) as (reservation, start):
        timer = perf_counter()
        result = cross_fit_splits(len(labels), splits, fit_fold)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            Trial(hypothesis, block, "TF-IDF logistic regression", "official-v1", {"block": block}),
            labels,
            splits,
            result,
            perf_counter() - timer,
            lambda _: None,
            save_query=False,
            gate=(source, start, (len(labels), len(CLASSES))),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("eda", "cv", "select"))
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--plan", type=Path, default=COMP / "selection-plan.json")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action == "cv":
        if not args.trial or not args.run_id:
            parser.error("--trial と --run-id が必要です")
        run(args.trial, args.run_id, args.plan.resolve())
        return
    frame, labels, groups, splits, evaluation = load_cv()
    if args.action == "eda":
        print(
            json.dumps(
                {
                    "rows": len(frame),
                    "essays": frame[ESSAY].nunique(),
                    "groups": len(set(groups)),
                    "counts": np.bincount(labels).tolist(),
                    "role_rates": pd.crosstab(frame[ROLE], labels, normalize="index")
                    .round(4)
                    .to_dict(orient="index"),
                    "words": np.quantile(
                        [len(text.split()) for text in frame[TEXT]], [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                    "rows_per_essay": frame.groupby(ESSAY)
                    .size()
                    .quantile([0, 0.25, 0.5, 0.75, 1])
                    .tolist(),
                    "fold_sizes": [len(valid) for _, valid in splits],
                    "split_hash": evaluation.split_hash,
                }
            )
        )
        return
    if not args.receipt:
        parser.error("--receipt が必要です")
    store = RunStore(COMP)
    plan = read_plan(args.plan)
    check_candidates(store, plan)
    result = select_runs(store, plan, assess_for(labels, splits, evaluation))
    write_selection(args.receipt, result)
    print(result)


if __name__ == "__main__":
    main()
