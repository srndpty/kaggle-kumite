"""手法の定義と、ある締切での学習・予測。CV と最終提出が同じ入口を使う。"""

from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
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


def training_rows(
    tables: Tables, rule: Rule, label_start: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """label_start からの 7 日間の購入者について、その締切の候補・特徴と購入の有無を作る。

    候補内に正解のない顧客は順位学習に寄与しないので除く。
    """
    truth = window_truth(tables.transactions, label_start)
    snapshot = Snapshot.at(tables.transactions, tables.articles, label_start, rule)
    candidates = snapshot.candidates(truth.customers)
    ranked = candidate_matrix(candidates, truth.customers, rule.width, tables.n_articles)
    labels = sparse_hits(ranked, truth.rows, truth.items, tables.n_articles)
    keep = labels.any(axis=1)
    kept = candidates.filter(np.repeat(keep, rule.width))
    x = FeatureBuilder(tables, snapshot).matrix(kept)
    return x, labels[keep].ravel(), int(keep.sum())


def fit(tables: Tables, method: Method, cutoff: int) -> lgb.Booster:
    """cutoff の直前から label_weeks 週ぶんの購入を教師にする。"""
    parts = [
        training_rows(tables, method.rule, cutoff - HORIZON * week)
        for week in range(method.label_weeks, 0, -1)
    ]
    data = lgb.Dataset(
        np.concatenate([x for x, _, _ in parts]),
        np.concatenate([y for _, y, _ in parts]).astype(np.int8),
        group=np.full(sum(n for _, _, n in parts), method.rule.width),
        feature_name=feature_names(method.rule),
    )
    return lgb.train(PARAMS, data, num_boost_round=ROUNDS)


def recommend(
    tables: Tables, method: Method, cutoff: int, customers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, lgb.Booster | None]:
    """cutoff より前の取引だけから、各顧客の上位 K 件と候補 width 件を返す。customers は昇順。"""
    past = Snapshot.at(tables.transactions, tables.articles, cutoff, method.rule)
    visible = Tables(
        past.history, tables.articles, tables.ages, tables.customer_ids, tables.article_ids
    )
    model = fit(visible, method, cutoff) if method.label_weeks else None
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
    return np.concatenate(top), np.concatenate(pool), model
