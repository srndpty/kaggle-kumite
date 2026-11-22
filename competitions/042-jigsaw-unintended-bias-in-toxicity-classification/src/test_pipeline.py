"""指標の部分集合、分割、提出形式の回帰テスト。"""

import numpy as np
import pandas as pd
import pytest
from dataset import labels_and_groups, make_folds, submission
from metrics import IDENTITIES, breakdown, subsets
from model import soft_binary_examples
from select_model import bootstrap, rank_auc
from sklearn.metrics import roc_auc_score

from kumite.text_features import build_features


def test_bias_auc_subsets_and_aggregate() -> None:
    target = [False, True, False, True]
    membership = [True, True, False, False]
    labels = np.array([target, *([membership] * len(IDENTITIES))]).T
    masks = [mask.tolist() for _, _, mask in list(subsets(labels))[1:4]]
    assert masks == [
        [True, True, False, False],
        [True, False, False, True],
        [False, True, True, False],
    ]
    result = breakdown(labels, np.array([0.6, 0.5, 0.1, 0.9]))
    assert result["overall_auc"] == pytest.approx(0.75)
    assert result["subgroup_power_mean"] == 0
    assert result["bpsn_power_mean"] == pytest.approx(1)
    assert result["bnsp_power_mean"] == pytest.approx(1)
    assert result["score"] == pytest.approx(0.6875)


def test_grouped_folds_include_all_rows() -> None:
    train = pd.DataFrame(
        [
            {
                "comment_text": f" Text   {i} ",
                "target": float(i % 2),
                **dict.fromkeys(IDENTITIES, 1.0 if i % 4 < 2 else np.nan),
            }
            for i in range(90)
        ]
    )
    train = pd.concat([train, train.iloc[[0]].assign(comment_text="text 0")], ignore_index=True)
    labels, groups = labels_and_groups(train)
    folds = make_folds(labels, groups)
    assert folds[0] == folds[-1]
    assert set(folds) == {0, 1, 2}
    assert not labels[2, 1:].any()


def test_submission_orders_by_official_sample() -> None:
    test = pd.DataFrame({"id": [8, 4]})
    sample = pd.DataFrame({"id": [4, 8], "prediction": [0, 0]})
    result = submission(test, sample, np.array([0.1, 0.9]))
    assert result["id"].tolist() == [4, 8]
    assert result["prediction"].tolist() == [0.9, 0.1]
    with pytest.raises(ValueError, match="範囲外"):
        submission(test, sample, np.array([0.1, 1.1]))


def test_bigram_features_keep_baseline_columns() -> None:
    train = ["first phrase", "first phrase", "first phrase", "other phrase"]
    valid = ["first phrase"]
    query = ["first queryexclusive"]
    base, _ = build_features(train, valid, query)
    added, vectorizers = build_features(train, valid, query, additional_bigrams=True)
    assert vectorizers[1].get_feature_names_out().tolist() == ["first phrase"]
    for before, after in zip(base, added, strict=True):
        expected = before.toarray()
        assert np.array_equal(expected, after.toarray()[:, : expected.shape[1]])


def test_soft_labels_preserve_positive_mass_and_total_weight() -> None:
    target = np.array([0, 0.25, 0.5, 0.9, 1])
    rows, outcomes, weights = soft_binary_examples(target)
    assert len(rows) == 8
    np.testing.assert_allclose(np.bincount(rows, weights=weights), np.ones(len(target)))
    np.testing.assert_allclose(np.bincount(rows, weights=weights * outcomes), target)


def test_bootstrap_weighted_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(42)
    predictions = rng.integers(5, size=100)
    labels = rng.integers(2, size=(100, 1 + len(IDENTITIES))).astype(bool)
    groups = np.repeat(np.arange(50), 2)
    weights = rng.integers(10, size=50)
    expected = [
        roc_auc_score(labels[mask, 0], predictions[mask], sample_weight=weights[groups[mask]])
        for _, _, mask in subsets(labels)
    ]
    np.testing.assert_allclose(rank_auc(labels, groups, predictions)(weights), expected)


def test_paired_bootstrap_for_identical_predictions() -> None:
    target = np.resize([False, True, False, True], 80)
    membership = np.resize([False, False, True, True], 80)
    labels = np.array([target, *([membership] * len(IDENTITIES))]).T
    predictions = np.linspace(0, 1, 80)
    assert bootstrap(labels, np.arange(80), predictions, predictions, count=5) == (0, 0)
