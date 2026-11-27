"""セッション内にない人気候補の共通特徴量をまとめて採点する。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from candidates import Method, candidate_aids, rank_features
from evaluation import Event
from ranker import PARAMS, Ranker


@dataclass
class CachedRanker:
    ranker: Ranker
    method: Method
    popularity: Mapping[int, int]
    _defaults: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    def _score(self, features: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.ranker.model.predict(features, num_threads=PARAMS["num_threads"]))
        if scores.shape != (len(features),) or not np.isfinite(scores).all():
            raise ValueError("再ランキングの予測形状または値が不正です")
        return scores

    def _default_scores(self, length: int) -> np.ndarray:
        if length not in self._defaults:
            n_items = len(self.popularity)
            width = self.method.width
            features = np.zeros(
                (width * n_items, 6 + self.method.elapsed_feature), dtype=np.float32
            )
            features[:, 0] = np.repeat(np.arange(width), n_items)
            features[:, 1] = length
            features[:, 2] = np.tile(np.array(list(self.popularity.values())), width)
            if self.method.elapsed_feature:
                features[:, 6] = -1
            self._defaults[length] = self._score(features).reshape(width, n_items)
        return self._defaults[length]

    def rank(
        self, batch: Sequence[tuple[int, Sequence[Event]]]
    ) -> list[tuple[int, list[int], list[int]]]:
        popular = list(self.popularity)
        popular_index = {aid: index for index, aid in enumerate(popular)}
        candidates: list[list[int]] = []
        known_sizes: list[int] = []
        known_features: list[list[float]] = []
        for _, prefix in batch:
            aids = candidate_aids(prefix, popular, self.method.width)
            known = min(len({event["aid"] for event in prefix}), len(aids))
            candidates.append(aids)
            known_sizes.append(known)
            known_features.extend(
                rank_features(
                    prefix,
                    aids[:known],
                    self.popularity,
                    elapsed_feature=self.method.elapsed_feature,
                )
            )
        known_scores = self._score(np.asarray(known_features, dtype=np.float32))
        offset = 0
        result: list[tuple[int, list[int], list[int]]] = []
        for (session, prefix), aids, known in zip(batch, candidates, known_sizes, strict=True):
            stop = offset + known
            scores = known_scores[offset:stop]
            if known < len(aids):
                default = self._default_scores(len(prefix))
                ranks = np.arange(known, len(aids))
                items = [popular_index[aid] for aid in aids[known:]]
                scores = np.concatenate((scores, default[ranks, items]))
            top = np.argsort(-scores, kind="stable")[:20]
            result.append((session, [aids[i] for i in top], aids))
            offset = stop
        return result
