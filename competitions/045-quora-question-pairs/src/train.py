"""事前登録した質問ペア実験を実行する。"""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from features import EXPERIMENTS, PairFeatures
from inputs import official_data_hash
from predict import ROOT, TEXT, read_test, write_submission
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from split import make_splits, question_nodes

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.runs import RunStore

HYPOTHESES = {
    "E01": "単語の類似度と質問の長さで重複を予測できる",
    "E02": "文字 n-gram が表記差に頑健",
    "E03": "単語包含率が追加条件を識別する",
    "E04": "数字と否定の不一致が紛らわしい非重複を識別する",
    "E05": "先頭・末尾の単語列の一致が質問の構造の差を捉える",
}


def pair_text(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [frame[col].fillna("").str.normalize("NFKC").str.casefold().to_numpy() for col in TEXT]
    )


def fit_model(
    pairs: np.ndarray, y: np.ndarray, experiment: str, seed: int
) -> tuple[PairFeatures, Pipeline]:
    features = PairFeatures(experiment)
    features.fit(pairs)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=500))
    model.fit(features.transform(pairs), y)
    return features, model


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y, pred, labels=[0, 1]))


def train(
    run_id: str, experiment: str, seed: int, root: Path = ROOT, *, cv_only: bool = False
) -> None:
    store = RunStore(root)
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        source = root / "data" / "train.csv"
        frame = pd.read_csv(source)
        labels = frame["is_duplicate"].to_numpy(dtype=np.int64)
        pairs = pair_text(frame)
        version, splits = make_splits(question_nodes(frame), labels, seed)
        roles = split_roles(len(frame), splits)
        ids = None
        query = np.empty((0, 2), dtype=object)
        if not cv_only:
            ids, test = read_test(root)
            query = pair_text(test)
            del test

        def fit_fold(train_rows: np.ndarray, valid: np.ndarray) -> FoldOutput:
            features, model = fit_model(pairs[train_rows], labels[train_rows], experiment, seed)
            valid_pred = model.predict_proba(features.transform(pairs[valid]))
            validate_probabilities(valid_pred, len(valid), 2)
            test_pred = np.empty(len(query))
            for start in range(0, len(query), 50_000):
                batch = slice(start, start + 50_000)
                probabilities = model.predict_proba(features.transform(query[batch]))
                validate_probabilities(probabilities, len(query[batch]), 2)
                test_pred[batch] = probabilities[:, 1]
            print(f"検証 {len(valid)} 行の fold が完了", flush=True)
            return FoldOutput(
                valid_pred[:, 1],
                test_pred,
                {"train_rows": len(train_rows), "valid_rows": len(valid)},
            )

        result = cross_fit_splits(len(frame), splits, fit_fold)
        data_hash = official_data_hash(root)
        evaluation = Evaluation(
            root.name,
            "log_loss",
            False,
            score,
            data_hash,
            version,
            hashlib.sha256(roles.tobytes()).hexdigest(),
            seed,
        )
        trial = Trial(
            HYPOTHESES[experiment],
            experiment,
            "L2 ロジスティック回帰",
            "公式 CSV",
            {"experiment": experiment, "seed": seed, "folds": len(splits), "cv_only": cv_only},
        )

        def write(out: Path) -> None:
            if ids is not None:
                write_submission(ids, result.query, out / "submission.csv")
            np.savez_compressed(out / "split.npz", ids=frame["id"].to_numpy(), roles=roles)

        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            labels,
            splits,
            result,
            perf_counter() - start,
            write,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-only", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment or args.run_id
    if experiment not in EXPERIMENTS:
        parser.error("--experiment を指定してください")
    train(args.run_id, experiment, args.seed, cv_only=args.cv_only)


if __name__ == "__main__":
    main()
