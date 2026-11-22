"""固定分割で学習し、OOF、採点内訳、提出ファイルを保存する。"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
from dataset import (
    FILES,
    SEED,
    SPLIT_VERSION,
    labels_and_groups,
    load_query,
    load_train,
    make_folds,
    submission,
    train_content_hash,
    validate,
    validate_folds,
)
from evidence import METRIC, notebook_sources, source_hashes
from joblib import Parallel, delayed
from metrics import IDENTITIES, breakdown, score
from model import MODEL_PARAMS, predict_fold

from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "E00": ("定数 0.5", "公式採点経路を確認する"),
    "E01": ("単語 unigram TF-IDF", "本文の単語で有害性を順位付けできる"),
    "E02": ("単語 bigram を追加", "短い文脈を拾うと BPSN が改善する"),
    "E03": ("文字 3–5 gram を追加", "綴り崩しを拾うと全体 AUC と BNSP が改善する"),
    "E04": ("属性言及ありの陰性の重みを2倍", "BPSN が改善し、BNSP が悪化する可能性がある"),
    "E05": ("有害判定の投票比率を用いた損失", "Subgroup AUC のべき平均が改善する"),
}


def profile(train, labels: np.ndarray, groups: np.ndarray) -> dict:
    duplicates = np.bincount(groups)[groups] > 1
    positive = np.bincount(groups, weights=labels[:, 0])
    negative = np.bincount(groups, weights=~labels[:, 0])
    return {
        "rows": len(train),
        "toxic": int(labels[:, 0].sum()),
        "duplicate_rows": int(duplicates.sum()),
        "conflicting_duplicate_groups": int(((positive > 0) & (negative > 0)).sum()),
        "identities": {
            identity: {
                "missing_rate": float(train[identity].isna().mean()),
                "positive": int((labels[:, index] & labels[:, 0]).sum()),
                "negative": int((labels[:, index] & ~labels[:, 0]).sum()),
            }
            for index, identity in enumerate(IDENTITIES, start=1)
        },
    }


def baseline_folds(
    store: RunStore,
    baseline: str,
    data_hash: str,
    train: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    record = store.record(baseline)
    if (
        record.params.get("trial") != "E01"
        or record.data_hash != data_hash
        or record.split_version != SPLIT_VERSION
    ):
        raise ValueError("公式データと分割設定が一致する E01 が必要です")
    path = store.artifact_path(baseline, "folds.csv")
    frame = pd.read_csv(path, dtype={"fold": "int8"})
    folds = frame["fold"].to_numpy()
    if not frame["id"].equals(train["id"]) or array_sha256(folds) != record.split_hash:
        raise ValueError("基準 run の行順または分割ハッシュが一致しません")
    validate_folds(folds, labels, groups)
    return folds, store.input_hashes(baseline, "folds.csv")


def run(root: Path, trial: str, run_id: str, baseline: str = "E01", jobs: int = 1) -> Path:
    data = root / "data"
    store = RunStore(root)
    change, hypothesis = TRIALS[trial]
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        train = load_train(data)
        test, sample = load_query(data)
        data_hash = files_sha256(data / name for name in FILES)
        if trial == "E00":
            labels = train[["target", *IDENTITIES]].to_numpy() >= 0.5
            diagnostics = breakdown(labels, np.full(len(train), 0.5))
            frame = submission(test, sample, np.full(len(test), 0.5))
            return store.publish(
                reservation,
                ExperimentRecord(
                    competition_id="042",
                    run_id=run_id,
                    metric=METRIC,
                    greater_is_better=True,
                    data_version="official-v1",
                    data_hash=data_hash,
                    split_version="none",
                    split_hash="",
                    seed=SEED,
                    hypothesis=hypothesis,
                    changed_one_thing=change,
                    model="定数",
                    params={"prediction": 0.5, "evaluation": "train_only"},
                    fold_scores=[],
                    oof_score=diagnostics["score"],
                    code_version=code_version(REPO_ROOT),
                    runtime_sec=perf_counter() - start,
                ),
                lambda out: frame.to_csv(out / "submission.csv", index=False),
            )

        labels, groups = labels_and_groups(train)
        notebook_evidence = {
            "train_content_sha256": train_content_hash(train),
            "source_sha256": source_hashes(notebook_sources()),
        }
        if trial == "E01":
            folds, inputs = make_folds(labels, groups), {}
        else:
            folds, inputs = baseline_folds(store, baseline, data_hash, train, labels, groups)
        eda = profile(train, labels, groups)
        print(json.dumps(eda), flush=True)
        text = train["comment_text"].to_numpy()
        target = train["target"].to_numpy()
        query_text = test["comment_text"].to_numpy()

        def fit_fold(fold: int, training: np.ndarray, valid: np.ndarray) -> FoldOutput:
            print(f"{run_id}: fold {fold + 1}/3", flush=True)
            return FoldOutput(
                *predict_fold(
                    text[training],
                    labels[training],
                    target[training],
                    text[valid],
                    query_text,
                    trial,
                )
            )

        splits = folds_to_splits(folds)
        outputs = cast(
            Iterator[FoldOutput],
            Parallel(n_jobs=jobs, max_nbytes=None, return_as="generator")(
                delayed(fit_fold)(fold, training, valid)
                for fold, (training, valid) in enumerate(splits)
            ),
        )
        result = cross_fit_splits(len(labels), splits, lambda _training, _valid: next(outputs))
        if source_hashes(notebook_sources()) != notebook_evidence["source_sha256"]:
            raise ValueError("学習中に Notebook のソースが変更されました")
        validate(result.oof, len(train))
        frame = submission(test, sample, result.query)

        def write_artifacts(out: Path) -> None:
            pd_folds = train[["id"]].assign(fold=folds, group=groups)
            pd_folds.to_csv(out / "folds.csv", index=False)
            frame.to_csv(out / "submission.csv", index=False)
            (out / "eda.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
            (out / "notebook_evidence.json").write_text(
                json.dumps(notebook_evidence, indent=2), encoding="utf-8"
            )
            (out / "metrics.json").write_text(
                json.dumps(breakdown(labels, result.oof), indent=2), encoding="utf-8"
            )

        return publish_cv_run(
            store,
            reservation,
            Evaluation(
                "042", METRIC, True, score, data_hash, SPLIT_VERSION, array_sha256(folds), SEED
            ),
            Trial(
                hypothesis,
                change,
                "ロジスティック回帰",
                "official-v1",
                {**MODEL_PARAMS, "trial": trial, "min_df": 3, "sublinear_tf": True, "jobs": jobs},
                inputs,
            ),
            labels,
            splits,
            result,
            perf_counter() - start,
            write_artifacts,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--baseline", default="E01")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    print(run(ROOT, args.trial, args.run_id or args.trial, args.baseline, args.jobs))
