"""実験の定義・実行・保存。train.py・calibrate.py・stacking.py で共通に使う。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cv import SEED, SPLIT_VERSION
from dataset import CLASSES, COMPETITION_DIR, DATA_FILES
from features import TRANSFORMS, SupervisedFeature
from models import MODELS
from sklearn.metrics import log_loss

from kumite.checks import validate_probabilities
from kumite.crossfit import CrossFitResult, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import Reservation, RunStore

COMPETITION_ID = "001"
METRIC = "multi_logloss"
STORE = RunStore(COMPETITION_DIR)


@dataclass(frozen=True)
class Experiment:
    model: str
    hypothesis: str
    changed_one_thing: str
    transform: str = "raw"  # 行ごとの変換（features.TRANSFORMS）
    supervised: SupervisedFeature | None = None  # ラベルを使う特徴（外側 fold ごとに作る）
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def data_version(self) -> str:
        if self.supervised is None:
            return self.transform
        return f"{self.transform}+{self.supervised.name}"


def multi_logloss(y: np.ndarray | pd.Series, pred: np.ndarray) -> float:
    return float(log_loss(y, pred, labels=range(len(CLASSES))))


def _with_columns(x: pd.DataFrame, values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.concat([x, pd.DataFrame(values, columns=columns, index=x.index)], axis=1)


def fold_inputs(
    exp: Experiment,
    data: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    query: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1 fold の (学習, 検証, query) の入力を作る。ラベルを使う特徴には学習行のラベルだけを渡す。"""
    transform = TRANSFORMS[exp.transform]
    train, valid = data.iloc[train_idx], data.iloc[valid_idx]
    x_tr, x_va, x_q = transform(train), transform(valid), transform(query)
    if exp.supervised is None:
        return x_tr, x_va, x_q
    f_tr, f_rest = exp.supervised(train, y.iloc[train_idx], pd.concat([valid, query]))
    columns = exp.supervised.columns
    return (
        _with_columns(x_tr, f_tr, columns),
        _with_columns(x_va, f_rest[: len(valid)], columns),
        _with_columns(x_q, f_rest[len(valid) :], columns),
    )


def run_cv(
    exp: Experiment, data: pd.DataFrame, y: pd.Series, folds: np.ndarray, query: pd.DataFrame
) -> CrossFitResult:
    """folds で CV し、OOF 予測と query への予測（fold モデルの平均）を返す。"""
    fit_predict = MODELS[exp.model]

    def fit_fold(train_idx: np.ndarray, valid_idx: np.ndarray):
        x_tr, x_va, x_q = fold_inputs(exp, data, y, train_idx, valid_idx, query)
        return fit_predict(x_tr, y.iloc[train_idx], x_va, y.iloc[valid_idx], x_q, exp.params)

    return cross_fit(folds, fit_fold)


def save_run(
    reservation: Reservation,
    exp: Experiment,
    y: pd.Series,
    folds: np.ndarray,
    result: CrossFitResult,
    test_ids: pd.Series,
    runtime_sec: float,
    inputs: dict[str, str] | None = None,
) -> None:
    """予測を検証してからスコアを計算し、成果物と実験ログを保存する（kumite.runs.RunStore）。"""
    validate_probabilities(result.oof, len(y), len(CLASSES))
    validate_probabilities(result.query, len(test_ids), len(CLASSES))
    evaluation = Evaluation(
        competition_id=COMPETITION_ID,
        metric=METRIC,
        greater_is_better=False,
        score=multi_logloss,
        data_hash=files_sha256(DATA_FILES),
        split_version=SPLIT_VERSION,
        split_hash=array_sha256(folds),
        seed=SEED,
    )
    trial = Trial(
        hypothesis=exp.hypothesis,
        changed_one_thing=exp.changed_one_thing,
        model=exp.model,
        data_version=exp.data_version,
        params=exp.params,
        inputs=inputs or {},
    )

    def write_submission(out_dir: Path) -> None:
        submission = pd.DataFrame(result.query, columns=CLASSES)
        submission.insert(0, "id", test_ids.to_numpy())
        submission.to_csv(out_dir / "submission.csv", index=False)

    publish_cv_run(
        STORE,
        reservation,
        evaluation,
        trial,
        y.to_numpy(),
        folds_to_splits(folds),
        result,
        runtime_sec,
        write_submission,
    )


def load_predictions(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(OOF 予測, テスト予測) を、ログのハッシュと照合して返す。"""
    return STORE.load_array(run_id, "oof.npy"), STORE.load_array(run_id, "test.npy")
