"""同条件の予測だけを固定係数で平均し、別runとして保存する。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cv import make_folds
from dataset import DATA_FILES, LABELS, load_data, submission
from metrics import macro_auc, scores
from pipeline import EXPERIMENTS, STORE

from kumite.checks import validate_multilabel_probabilities
from kumite.crossfit import CrossFitResult, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256


def load_source(run_id, experiment, seed, data_hash, folds, train_ids, n_test):
    record = STORE.record(run_id)
    if (
        record.competition_id != "041"
        or record.metric != "macro_auc"
        or not record.greater_is_better
        or record.seed != seed
        or record.data_hash != data_hash
        or record.split_hash != array_sha256(folds)
        or record.split_version != f"text-group-multilabel-3fold-v1-s{seed}"
        or record.params.get("experiment") != experiment
        or record.params.get("parent") != EXPERIMENTS[experiment][1]
        or record.params.get("labels") != list(LABELS)
    ):
        raise ValueError("元runのデータ・分割・実験定義が一致しません")
    saved_folds = pd.read_csv(STORE.artifact_path(run_id, "folds.csv"))
    if not np.array_equal(saved_folds.id.to_numpy(), train_ids) or not np.array_equal(
        saved_folds.fold.to_numpy(), folds
    ):
        raise ValueError("元runのfold行順が不一致です")
    oof, test = STORE.load_array(run_id, "oof.npy"), STORE.load_array(run_id, "test.npy")
    validate_multilabel_probabilities(oof, len(folds), 6)
    validate_multilabel_probabilities(test, n_test, 6)
    return oof, test


def blend(run_id, seed=42):
    with STORE.reserve(run_id) as reservation:
        started = time.monotonic()
        train, test, sample = load_data()
        y = train[list(LABELS)].to_numpy(dtype=np.int64)
        folds = make_folds(train.comment_text, y, seed)
        data_hash = files_sha256(DATA_FILES)
        suffix = "" if seed == 42 else f"-s{seed}"
        source_ids = ["E05" + suffix, "E03" + suffix]
        outputs = [
            load_source(r, e, seed, data_hash, folds, train.id.to_numpy(), len(test))
            for r, e in zip(source_ids, ("E05", "E03"), strict=True)
        ]
        result = CrossFitResult(
            oof=(outputs[0][0] + outputs[1][0]) / 2,
            query=(outputs[0][1] + outputs[1][1]) / 2,
            infos=[{"sources": source_ids, "weights": [0.5, 0.5]} for _ in range(3)],
        )
        inputs = {}
        for r in source_ids:
            inputs.update(STORE.input_hashes(r, "oof.npy", "test.npy", "folds.csv"))

        def write(out: Path):
            submission(sample, test.id.to_numpy(), result.query).to_csv(
                out / "submission.csv", index=False
            )
            pd.DataFrame({"id": train.id, "fold": folds}).to_csv(out / "folds.csv", index=False)
            metrics = {
                "oof": scores(y, result.oof),
                "folds": [scores(y[folds == k], result.oof[folds == k]) for k in range(3)],
            }
            (out / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        hypothesis, parent, change = EXPERIMENTS["E06"]
        publish_cv_run(
            STORE,
            reservation,
            Evaluation(
                "041",
                "macro_auc",
                True,
                macro_auc,
                data_hash,
                f"text-group-multilabel-3fold-v1-s{seed}",
                array_sha256(folds),
                seed,
            ),
            Trial(
                hypothesis,
                change,
                "fixed_mean",
                "official-v1",
                {
                    "experiment": "E06",
                    "parent": parent,
                    "labels": LABELS,
                    "sources": ["E05", "E03"],
                    "weights": [0.5, 0.5],
                },
                inputs,
            ),
            y,
            folds_to_splits(folds),
            result,
            time.monotonic() - started,
            write,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    blend(a.run_id, a.seed)
