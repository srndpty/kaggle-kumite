"""実験の事前定義、CVと成果物の保存。"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cv import make_folds
from dataset import COMPETITION_DIR, DATA_FILES, LABELS, load_data, submission
from metrics import macro_auc, scores
from models import fit_linear
from transformer import fit_transformer, prepare

from kumite.checks import validate_multilabel_probabilities
from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore
from kumite.text_features import build_features

STORE = RunStore(COMPETITION_DIR)
EXPERIMENTS = {
    "E00": ("陽性率の定数予測を基準にする", "", "定数予測"),
    "E01": ("単語で有効なbaselineになる", "E00", "単語TF-IDFとLR"),
    "E02": ("短い文脈を拾い改善する", "E01", "単語bigram追加"),
    "E03": ("綴りの変形に強くなる", "E01", "文字3-5gram追加"),
    "E04": ("重み付けで順位と確率への影響が異なる", "E01", "balanced重み"),
    "E05": ("事前学習Transformerが文脈を拾う", "E01", "DistilBERTに変更"),
    "E06": ("異なる表現の固定平均はE05を改善する", "E05", "E03との1:1確率平均"),
}


def train(run_id, experiment, seed=42, epochs=2):
    if experiment == "E06":
        raise ValueError("E06はblend.pyで実行してください")
    hypothesis, parent, change = EXPERIMENTS[experiment]
    with STORE.reserve(run_id) as reservation:
        started = time.monotonic()
        train_df, test_df, sample = load_data()
        y = train_df[list(LABELS)].to_numpy(dtype=np.int64)
        texts = train_df.comment_text.to_numpy()
        test_texts = test_df.comment_text.to_numpy()
        folds = make_folds(texts, y, seed)
        inputs = {}
        params = {
            "experiment": experiment,
            "parent": parent,
            "epochs": epochs,
            "labels": LABELS,
            "C": 1,
            "min_df": 3,
            "max_features_word": 100000,
            "max_features_char": 150000,
            "sublinear_tf": True,
            "solver": "liblinear",
            "max_iter": 1000,
            "weighted": experiment == "E04",
        }
        encoded = encoded_test = None
        if experiment == "E05":
            encoded, encoded_test, identity = prepare(texts, test_texts)
            inputs.update(identity)
            params.update(
                {
                    "model": "distilbert/distilbert-base-uncased",
                    "max_length": 256,
                    "batch_size": 32,
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                    "warmup": 0.1,
                }
            )

        def fit(tr, va):
            if experiment == "E00":
                prior = y[tr].mean(axis=0)
                return FoldOutput(np.tile(prior, (len(va), 1)), np.tile(prior, (len(test_df), 1)))
            if experiment == "E05":
                assert encoded is not None and encoded_test is not None
                return fit_transformer(encoded, encoded_test, y, tr, va, seed, epochs)
            frames, _ = build_features(
                texts[tr],
                texts[va],
                test_texts,
                bigram=experiment == "E02",
                chars=experiment == "E03",
            )
            valid, test = fit_linear(frames, y[tr], experiment == "E04")
            print(f"fold {int(folds[va[0]])}: {macro_auc(y[va], valid):.6f}", flush=True)
            return FoldOutput(valid, test)

        result = cross_fit(folds, fit)
        for pred in (result.oof, result.query):
            validate_multilabel_probabilities(pred, len(pred), len(LABELS))
        report = {
            "oof": scores(y, result.oof),
            "folds": [scores(y[folds == k], result.oof[folds == k]) for k in range(3)],
        }

        def write(out: Path):
            submission(sample, test_df.id.to_numpy(), result.query).to_csv(
                out / "submission.csv", index=False
            )
            pd.DataFrame({"id": train_df.id, "fold": folds}).to_csv(out / "folds.csv", index=False)
            (out / "metrics.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        publish_cv_run(
            STORE,
            reservation,
            Evaluation(
                "041",
                "macro_auc",
                True,
                macro_auc,
                files_sha256(DATA_FILES),
                f"text-group-multilabel-3fold-v1-s{seed}",
                array_sha256(folds),
                seed,
            ),
            Trial(
                hypothesis,
                change,
                "distilbert"
                if experiment == "E05"
                else "tfidf_lr"
                if experiment != "E00"
                else "prior",
                "official-v1",
                params,
                inputs,
            ),
            y,
            folds_to_splits(folds),
            result,
            time.monotonic() - started,
            write,
        )
