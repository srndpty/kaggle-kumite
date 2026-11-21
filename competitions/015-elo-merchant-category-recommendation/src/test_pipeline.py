"""特徴量・分割・提出形式の検査。"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from dataset import ID, TARGET, TRANSACTION_FILES, features, submission
from train import make_folds, rmse

from kumite.crossfit import folds_to_splits, split_roles


def test_features_keep_card_order_and_aggregate(tmp_path):
    history = pl.DataFrame(
        {
            ID: ["a", "a", "b"],
            "purchase_amount": [1.0, 3.0, 8.0],
            "authorized_flag": ["Y", "N", "Y"],
            "merchant_id": ["one", "one", "two"],
            "month_lag": [0, -1, -1],
        }
    )
    history.write_csv(tmp_path / TRANSACTION_FILES[0])
    cards = pd.DataFrame(
        {
            ID: ["b", "c", "a"],
            "feature_1": [1, 2, 3],
            "feature_2": [1, 2, 3],
            "feature_3": [1, 2, 3],
            "first_active_month": ["2017-01", None, "2018-02"],
        }
    )
    result = features(tmp_path, cards, "E02")
    assert ID not in result
    assert result["history_count"].tolist() == [1, 0, 2]
    assert result["recent_count"].tolist() == [0, 0, 1]
    assert result.loc[2, "history_merchants"] == 1
    assert result.loc[2, "history_amount_mean"] == 2
    assert result.loc[2, "history_approval_rate"] == 0.5


def test_folds_and_rmse():
    target = np.tile([-33.0, 1.0, 3.0], 10)
    folds = make_folds(target, 42)
    assert np.array_equal(folds, make_folds(target, 42))
    splits = folds_to_splits(folds)
    assert len(splits) == 5
    assert ((split_roles(len(target), splits) == 2).sum(axis=0) == 1).all()
    assert all((target[valid] < -20).sum() == 2 for _, valid in splits)
    assert rmse(np.array([2, 4]), np.array([2, 0])) == pytest.approx(np.sqrt(8))


def test_submission_uses_sample_order_and_rejects_invalid_predictions():
    test = pd.DataFrame({ID: ["a", "b"]})
    sample = pd.DataFrame({ID: ["b", "a"], TARGET: [0, 0]})
    result = submission(test, sample, np.array([1.0, -2.0]))
    assert list(result.columns) == [ID, TARGET]
    assert result.to_dict(orient="list") == {ID: ["b", "a"], TARGET: [-2.0, 1.0]}
    with pytest.raises(ValueError):
        submission(test, sample, np.array([1.0, np.inf]))
