"""行動別 LambdaRank。教師に入れるグループは候補幅と独立に指定する。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from candidates import Method, candidate_aids, rank_features, training_labels
from evaluation import Event, split_session

TRAIN_EVERY = 100
PREDICTION_BATCH = 256
PARAMS = {
    "objective": "lambdarank",
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "num_threads": 4,
    "deterministic": True,
    "force_col_wise": True,
    "seed": 42,
    "verbosity": -1,
}
ROUNDS = 100


@dataclass(frozen=True)
class Ranker:
    model: lgb.Booster
    groups: int
    rows: int
    positives: int


def fit_ranker(
    history: Iterable[tuple[int, list[Event]]],
    action: str,
    popular: Mapping[int, int],
    method: Method,
) -> Ranker:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    aids_by_popularity = list(popular)
    for session, events in history:
        prefix, truth = split_session(session, events)
        candidates = candidate_aids(prefix, aids_by_popularity, method.width)
        group_labels = training_labels(candidates, truth[action], method)
        if group_labels is None:
            continue
        labels.extend(group_labels)
        groups.append(len(candidates))
        features.append(
            np.asarray(
                rank_features(prefix, candidates, popular, elapsed_feature=method.elapsed_feature),
                dtype=np.float32,
            )
        )
    if not groups:
        raise ValueError(f"{action} の教師グループがありません")
    dataset = lgb.Dataset(
        np.concatenate(features), label=np.asarray(labels), group=np.asarray(groups)
    )
    model = lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)
    return Ranker(model, len(groups), len(labels), sum(labels))
