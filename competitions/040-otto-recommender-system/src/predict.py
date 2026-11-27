"""セッションごとに3行動の候補と上位20件を返す。"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import batched

from cached_inference import CachedRanker
from candidates import Method, candidate_aids
from evaluation import ACTIONS, Event
from ranker import PREDICTION_BATCH, Ranker

Predictions = Iterator[tuple[int, str, list[int], list[int]]]


def predict(
    prefixes: Iterable[tuple[int, Sequence[Event]]],
    method: Method,
    popularity: Mapping[str, dict[int, int]],
    rankers: Mapping[str, Ranker],
) -> Predictions:
    if not rankers:
        popular_aids = {action: list(popularity[action]) for action in ACTIONS}
        for session, prefix in prefixes:
            for action in ACTIONS:
                aids = candidate_aids(prefix, popular_aids[action], 20)
                yield session, action, aids, aids
        return

    scorers = [CachedRanker(rankers[action], method, popularity[action]) for action in ACTIONS]
    for batch in batched(prefixes, PREDICTION_BATCH):
        ranked = [scorer.rank(batch) for scorer in scorers]
        for session_actions in zip(*ranked, strict=True):
            for action, (session, top, candidates) in zip(ACTIONS, session_actions, strict=True):
                yield session, action, top, candidates
