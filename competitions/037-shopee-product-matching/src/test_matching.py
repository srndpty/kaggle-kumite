import numpy as np
import pandas as pd
from matching import (
    Candidates,
    accepted,
    fit_text,
    mutual_edges,
    row_metrics,
    top_candidates,
    tune_threshold,
)
from split import assign_folds, components
from train import predict_fold

from kumite.checks import check_label_independence


def test_components_join_labels_transitively() -> None:
    rows = pd.DataFrame(
        {"label_group": [1, 2, 2, 3, 4, 5], "image": ["a", "a", "b", "b", "c", "d"]}
    )
    labels = components(rows)
    folds = assign_folds(labels, 3)
    assert len(set(labels[:4])) == 1
    assert len(set(folds[:4])) == 1
    assert len(set(folds)) == 3


def test_self_only_and_threshold_ties() -> None:
    candidates = Candidates(np.array([[1], [0], [0]]), np.array([[0.8], [0.8], [0.4]]))
    labels = np.array([1, 1, 2])
    baseline = row_metrics(candidates, labels, 1.0)
    np.testing.assert_allclose(baseline["f1"], [2 / 3, 2 / 3, 1])
    threshold = tune_threshold([(candidates, labels)])
    assert threshold == 0.8
    np.testing.assert_allclose(row_metrics(candidates, labels, threshold)["f1"], 1)


def test_identical_scores_preserve_posting_id_order_and_cap() -> None:
    titles = ["same title"] * 51
    candidates = top_candidates(fit_text(titles, words=False), titles)
    np.testing.assert_array_equal(candidates.indices[0], np.arange(1, 51))
    assert accepted(candidates, 0.9)[0].sum() == 49


def test_image_retrieval_adds_a_candidate_without_changing_text_top_k() -> None:
    titles = ["same title"] * 52
    embeddings = np.eye(52, dtype=np.float32)
    embeddings[51] = embeddings[0]
    hashes = np.arange(52)
    encoders = fit_text(titles, words=False)
    text = top_candidates(encoders, titles)
    union = top_candidates(encoders, titles, images=embeddings, hashes=hashes)
    assert 51 not in text.indices[0]
    assert 51 in union.indices[0, np.isfinite(union.scores[0])]
    np.testing.assert_array_equal(text.indices[0], union.indices[0, :50])


def test_mutual_filter_keeps_both_directions_only() -> None:
    source = np.array([0, 1, 1, 2, 3])
    target = np.array([1, 0, 2, 3, 1])
    np.testing.assert_array_equal(mutual_edges(source, target), [True, True, False, False, False])


def test_holdout_labels_do_not_change_reranking() -> None:
    labels = np.repeat(np.arange(8), 2)
    rows = pd.DataFrame(
        {
            "title": [f"product {label} standard model" for label in labels],
            "image_phash": [str(label) for label in labels],
            "component": labels,
        }
    )
    embeddings = np.eye(8, dtype=np.float32)[labels]

    def build(y: np.ndarray) -> list[np.ndarray]:
        labeled = rows.assign(label_group=y)
        result, threshold = predict_fold(labeled.iloc[:12], labeled.iloc[12:], "E03", embeddings)
        return [result.indices, result.scores, accepted(result, threshold)]

    perturbed = labels.copy()
    perturbed[12:] = np.arange(20, 24)
    check_label_independence(build, labels, perturbed)
