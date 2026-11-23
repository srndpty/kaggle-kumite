"""欠損処理の fold 境界を検証する。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from train import load, make_model, make_splits, transform


def test_median_uses_training_rows_only() -> None:
    train = pd.DataFrame({"number": [2.0, 4.0, None]})
    valid = pd.DataFrame({"number": [None, 1000.0]})
    _, result = transform(train, valid, "E02")
    assert result["number"].tolist() == [3.0, 1000.0]
    assert result["number_missing"].tolist() == [True, False]


@pytest.mark.parametrize("second", [None, "other"])
def test_missing_category_differs_from_unseen_category(second: str | None) -> None:
    train = pd.DataFrame({"category": ["known", second]})
    valid = pd.DataFrame({"category": [None, "unseen", "known"]})
    _, result = transform(train, valid, "E03")
    assert result["category"].cat.codes.tolist() == [len(train["category"].dropna()), -1, 0]


@pytest.mark.parametrize("variant", ["E00", "E01", "E02", "E03", "E04"])
def test_apply_rows_do_not_change_fit(variant: str) -> None:
    train = pd.DataFrame(
        {
            "v40": np.arange(100, dtype=float),
            "v50": [None, 2.0] * 50,
            "category": ["left", "right"] * 50,
        }
    )
    y = np.repeat([0, 1], 50)
    probe = train.iloc[[0, 99]]
    other = pd.DataFrame({"v40": [np.nan], "v50": [999.0], "category": ["new"]})
    first, first_probe = transform(train, probe, variant)
    second, second_probe = transform(train, pd.concat([probe, other]), variant)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(
        make_model().fit(first, y).predict_proba(first_probe),
        make_model().fit(second, y).predict_proba(second_probe.iloc[: len(probe)]),
    )


def test_duplicate_rows_share_a_fold() -> None:
    x = pd.DataFrame({"value": np.repeat(np.arange(30), 2)})
    y = np.tile([0, 1], 30)
    for train, valid in make_splits(x, y):
        assert not set(x.iloc[train]["value"]) & set(x.iloc[valid]["value"])


def test_sample_order_controls_query_order(tmp_path: Path) -> None:
    pd.DataFrame({"ID": [1, 2], "target": [0, 1], "value": [10, 20]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"ID": [3, 4], "value": [30, 40]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"ID": [4, 3], "PredictedProb": [0.5, 0.5]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    _, _, query, sample = load(tmp_path)
    submission = sample.assign(PredictedProb=query["value"].to_numpy() / 100)
    assert list(submission.itertuples(index=False, name=None)) == [(4, 0.4), (3, 0.3)]
