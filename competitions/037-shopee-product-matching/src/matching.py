"""候補検索と、出品単位の集合評価。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

TOP_K = 50
MAX_MATCHES = 50


@dataclass(frozen=True)
class Candidates:
    indices: np.ndarray
    scores: np.ndarray
    features: np.ndarray | None = None


def fit_text(train_titles: list[str], *, words: bool) -> tuple[TfidfVectorizer, ...]:
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    char.fit(train_titles)
    if words:
        word = TfidfVectorizer(ngram_range=(1, 2))
        word.fit(train_titles)
        return char, word
    return (char,)


def top_candidates(
    encoders: tuple[TfidfVectorizer, ...],
    gallery_titles: list[str],
    *,
    images: np.ndarray | None = None,
    hashes: np.ndarray | None = None,
) -> Candidates:
    if not gallery_titles:
        raise ValueError("検索対象のタイトルが空です")
    if images is not None and (
        hashes is None
        or len(images) != len(gallery_titles)
        or len(hashes) != len(gallery_titles)
        or not np.isfinite(images).all()
    ):
        raise ValueError("有効な画像埋め込みとハッシュを検索対象ごとに指定してください")
    text = encoders[0].transform(gallery_titles).astype(np.float32)
    word_matrix = (
        encoders[1].transform(gallery_titles).astype(np.float32) if len(encoders) > 1 else None
    )

    n_rows = len(gallery_titles)
    k = min(TOP_K, n_rows - 1)
    width = k if images is None else min(2 * k, n_rows - 1)
    indices = np.empty((n_rows, width), dtype=np.int64)
    scores = np.empty((n_rows, width), dtype=np.float32)
    features = None if images is None else np.empty((n_rows, width, 3), dtype=np.float32)
    for start in range(0, n_rows, 128):
        stop = min(start + 128, n_rows)
        similarities = (text[start:stop] @ text.T).toarray()
        if word_matrix is not None:
            similarities += (word_matrix[start:stop] @ word_matrix.T).toarray()
            similarities *= 0.5
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        if images is not None and features is not None and hashes is not None:
            image_scores = images[start:stop] @ images.T
            image_scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            image_order = np.argsort(-image_scores, axis=1, kind="stable")[:, :k]
            mask = np.zeros(similarities.shape, dtype=bool)
            np.put_along_axis(mask, order, True, axis=1)
            np.put_along_axis(mask, image_order, True, axis=1)
            similarities[~mask] = -np.inf
            order = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
            text_features = np.take_along_axis(similarities, order, axis=1)
            features[start:stop, :, 0] = np.where(np.isfinite(text_features), text_features, 0)
            image_features = np.take_along_axis(image_scores, order, axis=1)
            features[start:stop, :, 1] = np.where(np.isfinite(image_features), image_features, 0)
            features[start:stop, :, 2] = hashes[start:stop, None] == hashes[order]
        indices[start:stop] = order
        scores[start:stop] = np.take_along_axis(similarities, order, axis=1)
    return Candidates(indices, scores, features)


def truth(labels: np.ndarray, candidates: Candidates) -> tuple[np.ndarray, np.ndarray]:
    codes, groups = pd.factorize(labels)
    sizes = np.bincount(codes, minlength=len(groups))[codes]
    return (codes[:, None] == codes[candidates.indices]) & np.isfinite(candidates.scores), sizes


def pair_features(candidates: Candidates) -> np.ndarray:
    if candidates.features is None:
        raise ValueError("画像を含むペア特徴量が必要です")
    return candidates.features[np.isfinite(candidates.scores)]


def fit_reranker(folds: list[tuple[Candidates, np.ndarray]]) -> Booster:
    model = LGBMClassifier(n_estimators=150, random_state=42, n_jobs=4, verbosity=-1)
    model.fit(
        np.concatenate([pair_features(candidates) for candidates, _ in folds]),
        np.concatenate(
            [
                truth(labels, candidates)[0][np.isfinite(candidates.scores)]
                for candidates, labels in folds
            ]
        ),
    )
    return model.booster_


def rerank(candidates: Candidates, model: Booster) -> Candidates:
    if not candidates.scores.size:
        return candidates
    scores = np.full_like(candidates.scores, -np.inf)
    probabilities = np.asarray(model.predict(pair_features(candidates), num_threads=4))
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("ペアの確率が [0, 1] の有限値ではありません")
    scores[np.isfinite(candidates.scores)] = probabilities
    # posting_id 順を同点時の優先順とする。
    by_id = np.argsort(candidates.indices, axis=1, kind="stable")
    scores = np.take_along_axis(scores, by_id, axis=1)
    indices = np.take_along_axis(candidates.indices, by_id, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return Candidates(
        np.take_along_axis(indices, order, axis=1), np.take_along_axis(scores, order, axis=1)
    )


def accepted(candidates: Candidates, threshold: float) -> np.ndarray:
    above = candidates.scores >= threshold
    return above & (np.cumsum(above, axis=1) < MAX_MATCHES)


def mutual_edges(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """両方向に採用されたペアだけを残す。"""
    edges = pd.MultiIndex.from_arrays([source, target])
    return edges.isin(pd.MultiIndex.from_arrays([target, source]))


def tune_threshold(folds: list[tuple[Candidates, np.ndarray]]) -> float:
    """候補の追加による F1 の差分を、同じスコアごとに累積する。"""
    scores: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for candidates, labels in folds:
        relevant, sizes = truth(labels, candidates)
        relevant = relevant[:, : MAX_MATCHES - 1]
        count = np.arange(relevant.shape[1] + 1)
        hits = np.column_stack((np.ones(len(labels)), 1 + np.cumsum(relevant, axis=1)))
        f1 = 2 * hits / (1 + sizes[:, None] + count)
        scores.append(candidates.scores[:, : MAX_MATCHES - 1].ravel())
        deltas.append(np.diff(f1, axis=1).ravel())
    values = np.concatenate(scores).astype(np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    if not len(values):
        return 1.0
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    gains = np.cumsum(np.concatenate(deltas)[finite][order])[ends]
    best_gain = max(0.0, float(gains.max()))
    if best_gain <= 1e-10:
        return float(sorted_scores[0] + 1e-6)
    best = np.flatnonzero(gains >= best_gain - 1e-10)[0]
    return float(sorted_scores[ends[best]])


def row_metrics(candidates: Candidates, labels: np.ndarray, threshold: float) -> pd.DataFrame:
    relevant, sizes = truth(labels, candidates)
    chosen = accepted(candidates, threshold)
    count = relevant.sum(axis=1)
    found = np.minimum(count, MAX_MATCHES - 1) + 1
    return pd.DataFrame(
        {
            "f1": 2 * (1 + (chosen & relevant).sum(axis=1)) / (1 + chosen.sum(axis=1) + sizes),
            "candidate_recall": np.divide(
                count, sizes - 1, out=np.ones(len(sizes)), where=sizes > 1
            ),
            "oracle_f1": 2 * found / (found + sizes),
        }
    )
