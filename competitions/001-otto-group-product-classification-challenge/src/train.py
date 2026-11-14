"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/train.py E01
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from cv import N_SPLITS, SEED, SPLIT_VERSION, make_folds
from dataset import (
    CLASSES,
    COMPETITION_DIR,
    FEATURES,
    OUTPUT_DIR,
    TARGET,
    encode_target,
    load_test,
    load_train,
)
from models import MODELS
from sklearn.metrics import log_loss

from kumite.experiments import LOG_FILE_NAME, ExperimentRecord, append_record

COMPETITION_ID = "001"
DATA_VERSION = "raw"


@dataclass(frozen=True)
class Experiment:
    model: str
    hypothesis: str
    changed_one_thing: str
    params: dict[str, Any] = field(default_factory=dict)


LGBM_BASE = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": SEED,
    "num_boost_round": 10000,
    "early_stopping": 100,
}

EXPERIMENTS: dict[str, Experiment] = {
    "E00": Experiment(
        model="prior",
        hypothesis="学習 fold のクラス比を返すだけの予測。全実験の下限の目安",
        changed_one_thing="なし（統計的 baseline）",
    ),
    "E01": Experiment(
        model="lightgbm",
        hypothesis="生カウントの 93 特徴量そのままの GBDT。以降の比較の基準",
        changed_one_thing="E00 からモデルを LightGBM に変更（パラメータはほぼ既定値）",
        params=LGBM_BASE,
    ),
}


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    fit_predict = MODELS[exp.model]
    train, test = load_train(), load_test()
    x, y, x_te = train[FEATURES], encode_target(train[TARGET]), test[FEATURES]
    folds = make_folds(y)

    started = time.perf_counter()
    oof = np.zeros((len(x), len(CLASSES)))
    test_pred = np.zeros((len(x_te), len(CLASSES)))
    fold_scores: list[float] = []
    fold_infos: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        tr, va = folds != fold, folds == fold
        result = fit_predict(x[tr], y[tr], x[va], y[va], x_te, exp.params)
        oof[va] = result.valid_pred
        test_pred += result.test_pred / N_SPLITS
        fold_scores.append(float(log_loss(y[va], result.valid_pred, labels=range(len(CLASSES)))))
        fold_infos.append(result.info)
        print(f"fold {fold}: {fold_scores[-1]:.5f} {result.info}")
    runtime = time.perf_counter() - started
    oof_score = float(log_loss(y, oof, labels=range(len(CLASSES))))
    print(
        f"OOF logloss: {oof_score:.5f}（fold 平均 {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}）"
    )

    run_dir = OUTPUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof.npy", oof)
    np.save(run_dir / "test.npy", test_pred)
    submission = pd.DataFrame(test_pred, columns=CLASSES)
    submission.insert(0, "id", test["id"])
    submission.to_csv(run_dir / "submission.csv", index=False)
    (run_dir / "fold_info.json").write_text(json.dumps(fold_infos), encoding="utf-8")

    append_record(
        COMPETITION_DIR / LOG_FILE_NAME,
        ExperimentRecord(
            competition_id=COMPETITION_ID,
            run_id=run_id,
            data_version=DATA_VERSION,
            split_version=SPLIT_VERSION,
            seed=SEED,
            hypothesis=exp.hypothesis,
            changed_one_thing=exp.changed_one_thing,
            model=exp.model,
            params=exp.params,
            fold_scores=[round(s, 5) for s in fold_scores],
            oof_score=round(oof_score, 5),
            runtime_sec=round(runtime, 1),
        ),
    )
    print(f"出力: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=list(EXPERIMENTS))
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
