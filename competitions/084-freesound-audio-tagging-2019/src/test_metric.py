import numpy as np
import pytest
from sklearn.metrics import label_ranking_average_precision_score

from kumite.lwlrap import lwlrap


def test_lwlrap_weights_positive_labels_equally() -> None:
    labels = np.array([[True, False, True], [False, True, False], [False, False, False]])
    scores = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    assert lwlrap(labels, scores) == pytest.approx((1 + 2 / 3 + 1 / 2) / 3)


def test_lwlrap_matches_reference_tie_order() -> None:
    labels = np.array([[True, False]])
    assert lwlrap(labels, np.zeros((1, 2))) == 0.5
    labels = np.array([[True, False, True], [False, True, False]])
    assert lwlrap(labels, np.zeros((2, 3))) == (1 + 2 / 3 + 1 / 2) / 3


def test_lwlrap_matches_weighted_sklearn_reference_without_ties() -> None:
    rng = np.random.default_rng(42)
    labels = rng.random((100, 80)) < 0.04
    scores = rng.random(labels.shape)
    weights = labels.sum(axis=1)
    keep = weights > 0
    reference = label_ranking_average_precision_score(
        labels[keep], scores[keep], sample_weight=weights[keep]
    )
    assert lwlrap(labels, scores) == pytest.approx(reference)
