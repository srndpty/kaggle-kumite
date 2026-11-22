"""ローカル CV と Kaggle Notebook で共用する fold 内学習。"""

from typing import TypedDict, cast

import numpy as np
from dataset import SEED
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression

from kumite.text_features import build_features

Prediction = tuple[np.ndarray, np.ndarray, dict[str, int | float | bool | list[float]]]


class ModelParams(TypedDict):
    C: float
    max_iter: int
    tol: float
    random_state: int


MODEL_PARAMS: ModelParams = {"C": 1.0, "max_iter": 100, "tol": 0.001, "random_state": SEED}


def soft_binary_examples(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0 < target < 1 の行を二値2例に展開し、1行分の損失を分ける。"""
    probability = target.astype(np.float64)
    hard = probability >= 0.5
    fractional = np.flatnonzero((probability > 0) & (probability < 1))
    primary = np.where(hard, probability, 1 - probability)
    rows = np.concatenate((np.arange(len(target)), fractional))
    outcomes = np.concatenate((hard, ~hard[fractional]))
    weights = np.concatenate((primary, 1 - primary[fractional]))
    return rows, outcomes, weights


def predict_fold(
    text: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray,
    query: np.ndarray,
    trial: str,
) -> Prediction:
    (x_train, x_valid, x_test), _ = build_features(
        text, validation, query, additional_bigrams=trial == "E02", chars=trial == "E03"
    )
    weights = None
    outcomes = labels[:, 0]
    if trial == "E04":
        weights = 1.0 + (~outcomes & labels[:, 1:].any(axis=1))
        weights /= weights.mean()
    elif trial == "E05":
        rows, outcomes, weights = soft_binary_examples(target)
        x_train = vstack((x_train, x_train[rows[len(target) :]]), format="csr")
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(x_train, outcomes, sample_weight=weights)
    iterations = model.n_iter_
    if iterations is None:
        raise RuntimeError("学習の反復数を取得できません")
    valid = model.predict_proba(x_valid)[:, 1]
    info: dict[str, int | float | bool | list[float]] = {
        "iterations": int(iterations.max()),
        "features": int(np.size(model.coef_)),
    }
    if trial in {"E01", "E05"}:
        logits = model.decision_function(x_train)
        loss = binary_loss(logits, outcomes)
        info.update(
            coefficient_l2=float(np.linalg.norm(model.coef_)),
            training_bce=float(np.average(loss, weights=weights)),
            validation_quantiles=np.quantile(valid, [0.1, 0.5, 0.9]).tolist(),
            near_constant_validation=bool(np.ptp(np.quantile(valid, [0.1, 0.9])) < 0.01),
        )
    return (
        valid,
        model.predict_proba(x_test)[:, 1],
        info,
    )


def binary_loss(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, logits) - target * logits


def predict_ensemble(
    text: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    query: np.ndarray,
    trial: str,
    *,
    jobs: int = 1,
) -> np.ndarray:
    training_masks = (folds != fold for fold in range(3))
    results = cast(
        list[Prediction],
        Parallel(n_jobs=jobs, max_nbytes=None)(
            delayed(predict_fold)(
                text[training], labels[training], target[training], text[:1], query, trial
            )
            for training in training_masks
        ),
    )
    return np.mean([out for _, out, _ in results], axis=0)
