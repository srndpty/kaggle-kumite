"""グループ分割、順序予測、内側 OOF による閾値調整。"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from dataset import CATEGORIES
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from kumite.checks import validate_probabilities, validate_regression
from kumite.crossfit import CrossFitResult, FoldOutput, cross_fit, cross_fit_splits, folds_to_splits
from kumite.text_features import build_features

SEED = 42
DESCRIPTION_COMPONENTS = 16
LABELS = np.arange(5)
FIXED = np.array([0.5, 1.5, 2.5, 3.5])
PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    thread_count=4,
)


def qwk(y: np.ndarray, pred: np.ndarray) -> float:
    score = float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic"))
    if not np.isfinite(score):
        raise ValueError("QWK を計算できません")
    return score


def classify(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    validate_regression(scores, len(scores))
    return np.searchsorted(thresholds, scores, side="right")


def make_folds(y: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED)
    folds = np.full(len(y), -1)
    for fold, (fit, valid) in enumerate(splitter.split(y, y, groups)):
        if np.intersect1d(groups[fit], groups[valid]).size or set(y[fit]) != set(LABELS):
            raise ValueError("グループの交差または学習クラスの欠落があります")
        folds[valid] = fold
    if (folds < 0).any() or len(np.unique(folds)) != count:
        raise ValueError("fold 割当が不完全です")
    return folds


def fit_scores(
    x: pd.DataFrame, y: np.ndarray, query: pd.DataFrame, *, regression: bool
) -> np.ndarray:
    if regression:
        regressor = CatBoostRegressor(loss_function="RMSE", **PARAMS)
        regressor.fit(x, y, cat_features=list(CATEGORIES))
        scores = np.asarray(regressor.predict(query))
        validate_regression(scores, len(query))
        return scores
    classifier = CatBoostClassifier(loss_function="MultiClass", **PARAMS)
    classifier.fit(x, y, cat_features=list(CATEGORIES))
    probabilities = np.asarray(classifier.predict_proba(query))
    validate_probabilities(probabilities, len(query), len(LABELS))
    return probabilities @ classifier.classes_


def learn_thresholds(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    candidates = np.unique(np.r_[FIXED, np.quantile(scores, np.linspace(0, 1, 101))])
    thresholds = FIXED.copy()
    best = qwk(y, classify(scores, thresholds))
    while True:
        improved = False
        for index in range(len(thresholds)):
            lower, upper = np.r_[-np.inf, thresholds, np.inf][[index, index + 2]]
            chosen = thresholds[index]
            for candidate in candidates[(candidates > lower) & (candidates < upper)]:
                thresholds[index] = candidate
                score = qwk(y, classify(scores, thresholds))
                if score > best:
                    best, chosen = score, candidate
                    improved = True
            thresholds[index] = chosen
        if not improved:
            return thresholds


def inner_thresholds(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    folds = make_folds(y, groups, 3)

    def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        scores = fit_scores(x.iloc[train], y[train], x.iloc[valid], regression=False)
        return FoldOutput(scores, np.empty(0))

    return learn_thresholds(y, cross_fit(folds, fit).oof)


def fold_thresholds(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, fit: np.ndarray
) -> np.ndarray:
    return inner_thresholds(x.iloc[fit], y[fit], groups[fit])


def calibrate(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    query: np.ndarray,
) -> CrossFitResult:
    splits = folds_to_splits(folds)
    oof = np.column_stack([scores, np.zeros(len(scores))])
    infos = []
    with ProcessPoolExecutor(max_workers=len(splits)) as pool:
        jobs = [pool.submit(fold_thresholds, x, y, groups, fit) for fit, _ in splits]
        for (_, valid), job in zip(splits, jobs, strict=True):
            thresholds = job.result()
            oof[valid, 1] = classify(scores[valid], thresholds)
            infos.append({"thresholds": thresholds.tolist()})
    return CrossFitResult(oof, query, infos)


def predict_fold(
    x: pd.DataFrame,
    y: np.ndarray,
    fit: np.ndarray,
    valid: np.ndarray,
    test: pd.DataFrame,
    variant: str,
) -> FoldOutput:
    fit_x, fit_y = x.iloc[fit], y[fit]
    query = pd.concat([x.iloc[valid], test], ignore_index=True)
    if variant == "description":
        (fit_words, valid_words, test_words), _ = build_features(
            fit_x["Description"], x.iloc[valid]["Description"], test["Description"], bigram=True
        )
        svd = TruncatedSVD(n_components=DESCRIPTION_COMPONENTS, random_state=SEED)
        fit_components = svd.fit_transform(fit_words)
        query_components = np.vstack([svd.transform(valid_words), svd.transform(test_words)])
        columns = [f"description_svd_{i}" for i in range(fit_components.shape[1])]
        fit_x = fit_x.drop(columns="Description").join(
            pd.DataFrame(fit_components, index=fit_x.index, columns=columns)
        )
        query = query.drop(columns="Description").join(
            pd.DataFrame(query_components, index=query.index, columns=columns)
        )
    scores = fit_scores(fit_x, fit_y, query, regression=variant == "regression")
    valid_scores, test_scores = scores[: len(valid)], scores[len(valid) :]
    return FoldOutput(
        np.column_stack([valid_scores, classify(valid_scores, FIXED)]),
        test_scores,
        {"thresholds": FIXED.tolist()},
    )


def evaluate(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    test: pd.DataFrame,
    variant: str,
    *,
    workers: int = 1,
) -> CrossFitResult:
    def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        return predict_fold(x, y, train, valid, test, variant)

    if workers > 1:
        splits = folds_to_splits(folds)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {
                int(valid[0]): pool.submit(predict_fold, x, y, train, valid, test, variant)
                for train, valid in splits
            }
            return cross_fit_splits(len(y), splits, lambda _, valid: jobs[int(valid[0])].result())
    return cross_fit(folds, fit)
