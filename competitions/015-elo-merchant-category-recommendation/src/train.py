"""Elo の実験を学習し、分割・予測・提出ファイルを保存する。"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import pandas as pd
from dataset import CARD_FILES, ID, TARGET, TRANSACTION_FILES, features, load_cards, submission
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold

from kumite.checks import validate_regression
from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
EXTREME = -20.0
HYPOTHESES = {
    "E01": "履歴取引の全期間集約でロイヤルティを予測できる",
    "E02": "直近月の取引集約が全期間集約を補う",
    "E03": "新規加盟店の取引集約が履歴取引を補う",
    "E04": "極端値の発生確率を分けて学習すると全体 RMSE が下がる",
    "E05": "月別の取引件数で全期間の集約が捨てる活動パターンを補える",
}


class ModelParams(TypedDict):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    n_jobs: int
    random_state: int
    verbosity: int
    deterministic: bool
    force_col_wise: bool


PARAMS: ModelParams = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "n_jobs": 4,
    "random_state": 42,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def make_folds(y: np.ndarray, seed: int) -> np.ndarray:
    extreme = y < EXTREME
    if np.bincount(extreme, minlength=2).min() < 5:
        raise ValueError("極端値・通常値の各群に5件以上必要です")
    folds = np.empty(len(y), dtype=np.int8)
    for fold, (_, valid) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=seed).split(y, extreme)
    ):
        folds[valid] = fold
    return folds


def predict(x: pd.DataFrame, y: np.ndarray, query: pd.DataFrame, trial: str) -> np.ndarray:
    if trial != "E04":
        return np.asarray(LGBMRegressor(**PARAMS).fit(x, y).predict(query))
    extreme = y < EXTREME
    classifier = LGBMClassifier(**PARAMS).fit(x, extreme.astype(int))
    probability = np.asarray(classifier.predict_proba(query))[:, 1]
    validate_regression(probability, len(query), lower=0)
    if (probability > 1).any():
        raise ValueError("極端値の確率が1を超えています")
    low = LGBMRegressor(**PARAMS).fit(x.loc[extreme], y[extreme]).predict(query)
    normal = LGBMRegressor(**PARAMS).fit(x.loc[~extreme], y[~extreme]).predict(query)
    return probability * np.asarray(low) + (1 - probability) * np.asarray(normal)


def fit_fold(
    x: pd.DataFrame, y: np.ndarray, training: np.ndarray, valid: np.ndarray, trial: str
) -> FoldOutput:
    query = pd.concat([x.iloc[valid], x.iloc[len(y) :]])
    predicted = predict(x.iloc[training], y[training], query, trial)
    return FoldOutput(predicted[: len(valid)], predicted[len(valid) :])


def run(root: Path, trial: str, run_id: str, seed: int) -> Path:
    data = root / "data"
    store = RunStore(root)
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        train, test, sample = load_cards(data)
        y = train[TARGET].to_numpy()
        if trial == "E00":
            mean = float(y.mean())
            record = ExperimentRecord(
                competition_id="015",
                run_id=run_id,
                metric="RMSE",
                greater_is_better=False,
                data_version="official",
                data_hash=files_sha256(data / name for name in CARD_FILES),
                split_version="none",
                split_hash="",
                seed=seed,
                hypothesis="提出経路を確認する",
                changed_one_thing="学習全体の目的値平均",
                model="定数",
                params={"mean": mean, "evaluation": "train_only"},
                fold_scores=[],
                oof_score=rmse(y, np.full(len(y), mean)),
                code_version=code_version(ROOT.parents[1]),
            )
            frame = submission(test, sample, np.full(len(test), mean))
            return store.publish(
                reservation, record, lambda out: frame.to_csv(out / "submission.csv", index=False)
            )

        data_hash = files_sha256(data / name for name in (*CARD_FILES, *TRANSACTION_FILES))
        cards = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
        x = features(data, cards, trial)
        folds = make_folds(y, seed)

        result = cross_fit(folds, lambda training, valid: fit_fold(x, y, training, valid, trial))
        validate_regression(result.oof, len(train))
        frame = submission(test, sample, result.query)

        def write(out: Path) -> None:
            pd.DataFrame({ID: train[ID], "fold": folds}).to_csv(out / "folds.csv", index=False)
            frame.to_csv(out / "submission.csv", index=False)

        return publish_cv_run(
            store,
            reservation,
            Evaluation(
                "015",
                "RMSE",
                False,
                rmse,
                data_hash,
                f"card-extreme-v1-seed{seed}",
                array_sha256(folds),
                seed,
            ),
            Trial(
                HYPOTHESES[trial],
                trial,
                "LightGBM",
                "official-v1",
                {**PARAMS, "trial": trial, "extreme_threshold": EXTREME},
            ),
            y,
            folds_to_splits(folds),
            result,
            perf_counter() - start,
            write,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=["E00", *HYPOTHESES])
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run(ROOT, args.trial, args.run_id or args.trial, args.seed))
