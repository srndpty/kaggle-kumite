"""手法の定義と、ある締切での学習・予測。CV と最終提出が同じ入口を使う。"""

from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from candidates import Rule, Snapshot, candidate_matrix
from data import Tables
from evaluation import HORIZON, K, window_truth
from features import HISTORY_DAYS, FeatureBuilder, feature_names

from kumite.ranking import sparse_hits

PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "lambdarank_truncation_level": K,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 24,
    "verbosity": -1,
}
ROUNDS = 200
CHUNK = 50_000


@dataclass(frozen=True)
class Method:
    rule: Rule
    label_weeks: int = 0  # 0 なら規則の順位をそのまま使う
    label_pool: int | None = None  # 教師の顧客を、候補の先頭 label_pool 件に正解がある顧客に限る


METHODS = {
    "E00": Method(Rule(own=False, half_life=None, covisit=False)),
    "E01": Method(Rule(own=True, half_life=None, covisit=False)),
    "E02": Method(Rule(own=True, half_life=3.0, covisit=False)),
    "E03": Method(Rule(own=True, half_life=None, covisit=True)),
    "E04": Method(Rule(own=True, half_life=None, covisit=True), label_weeks=1),
    "E05": Method(Rule(own=True, half_life=None, covisit=True), label_weeks=4),
    "E06": Method(
        Rule(own=True, half_life=None, covisit=True, own_days=HISTORY_DAYS), label_weeks=1
    ),
    "E07": Method(Rule(own=True, half_life=None, covisit=True, product=True), label_weeks=1),
    "E08": Method(Rule(own=True, half_life=None, covisit=True, width=300), label_weeks=1),
    "E10": Method(
        Rule(own=True, half_life=None, covisit=True, width=300), label_weeks=1, label_pool=100
    ),
    "E09": Method(
        Rule(
            own=True,
            half_life=None,
            covisit=True,
            own_days=HISTORY_DAYS,
            product=True,
            width=300,
        ),
        label_weeks=1,
    ),
}


def settings(name: str) -> dict[str, object]:
    method = METHODS[name]
    return {
        "method": asdict(method),
        "params": PARAMS,
        "rounds": ROUNDS,
        "features": feature_names(method.rule),
    }


@dataclass(frozen=True)
class TrainingRows:
    x: np.ndarray
    y: np.ndarray
    groups: int
    stats: dict[str, float]


@dataclass(frozen=True)
class Recommendation:
    top: np.ndarray  # (顧客, K)
    pool: np.ndarray  # (顧客, 候補数)
    model: lgb.Booster | None
    training: list[dict[str, float]]  # 教師週ごとの診断


def label_candidates(
    tables: Tables, snapshot: Snapshot, label_pool: int | None
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    """snapshot の締切からの 7 日間の購入者の（候補, 購入の有無, 教師に使うか, 診断）。

    候補内に正解のない顧客は、正負の順位の組を作れないので教師から除く。候補の作り方を変えると
    教師の顧客も変わるため、その構成を診断として返す。
    """
    truth = window_truth(tables.transactions, snapshot.cutoff)
    candidates = snapshot.candidates(truth.customers)
    width = snapshot.rule.width
    ranked = candidate_matrix(candidates, truth.customers, width, tables.n_articles)
    labels = sparse_hits(ranked, truth.rows, truth.items, tables.n_articles)
    keep = labels[:, :label_pool].any(axis=1)
    stats = {
        "label_start": float(snapshot.cutoff),
        "buyers": float(len(keep)),
        "kept": float(keep.sum()),
        "excluded_share": float(1 - keep.mean()),
        "purchases_per_buyer": float(truth.n_relevant.mean()),
        "purchases_per_kept": float(truth.n_relevant[keep].mean()),
        "positives_per_kept": float(labels[keep].sum(axis=1).mean()),
        "positive_rate": float(labels[keep].mean()),
        "kept_with_own_candidate": float(
            (candidates["own_rank"].is_not_null().to_numpy().reshape(-1, width).any(axis=1))[
                keep
            ].mean()
        ),
    }
    return candidates, labels, keep, stats


def training_rows(tables: Tables, method: Method, label_start: int) -> TrainingRows:
    snapshot = Snapshot.at(tables.transactions, tables.articles, label_start, method.rule)
    candidates, labels, keep, stats = label_candidates(tables, snapshot, method.label_pool)
    kept = candidates.filter(np.repeat(keep, method.rule.width))
    x = FeatureBuilder(tables, snapshot).matrix(kept)
    return TrainingRows(x, labels[keep].ravel(), int(keep.sum()), stats)


def fit(tables: Tables, method: Method, cutoff: int) -> tuple[lgb.Booster, list[dict[str, float]]]:
    """cutoff の直前から label_weeks 週ぶんの購入を教師にする。"""
    parts = [
        training_rows(tables, method, cutoff - HORIZON * week)
        for week in range(method.label_weeks, 0, -1)
    ]
    data = lgb.Dataset(
        np.concatenate([part.x for part in parts]),
        np.concatenate([part.y for part in parts]).astype(np.int8),
        group=np.full(sum(part.groups for part in parts), method.rule.width),
        feature_name=feature_names(method.rule),
    )
    model = lgb.train(PARAMS, data, num_boost_round=ROUNDS)
    return model, [part.stats for part in parts]


def recommend(tables: Tables, method: Method, cutoff: int, customers: np.ndarray) -> Recommendation:
    """cutoff より前の取引だけから、各顧客の上位 K 件と候補 width 件を返す。customers は昇順。"""
    past = Snapshot.at(tables.transactions, tables.articles, cutoff, method.rule)
    visible = Tables(
        past.history, tables.articles, tables.ages, tables.customer_ids, tables.article_ids
    )
    model, training = fit(visible, method, cutoff) if method.label_weeks else (None, [])
    builder = FeatureBuilder(visible, past)
    top: list[np.ndarray] = []
    pool: list[np.ndarray] = []
    for begin in range(0, len(customers), CHUNK):
        part = customers[begin : begin + CHUNK]
        frame = past.candidates(part)
        matrix = candidate_matrix(frame, part, method.rule.width, tables.n_articles)
        pool.append(matrix)
        if model is None:
            top.append(matrix[:, :K])
            continue
        score = np.asarray(model.predict(builder.matrix(frame))).reshape(-1, method.rule.width)
        order = np.argsort(-score, axis=1, kind="stable")[:, :K]
        top.append(np.take_along_axis(matrix, order, axis=1))
    return Recommendation(np.concatenate(top), np.concatenate(pool), model, training)
