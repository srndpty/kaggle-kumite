"""特徴量・分割・提出形式の検査。"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from dataset import ID, TARGET, TRANSACTION_FILES, features, submission
from train import make_folds, rmse

from kumite.checks import check_label_independence
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


def test_e05_month_counts_handle_unknown_lag_and_missing_history(tmp_path):
    pl.DataFrame(
        {
            ID: ["a", "a", "a"],
            "purchase_amount": [1.0, 3.0, 8.0],
            "authorized_flag": ["Y", "N", "Y"],
            "merchant_id": ["one", "one", "two"],
            "month_lag": [0, -13, -14],
        }
    ).write_csv(tmp_path / TRANSACTION_FILES[0])
    cards = pd.DataFrame(
        {
            ID: ["a", "missing"],
            "feature_1": [1, 2],
            "feature_2": [1, 2],
            "feature_3": [1, 2],
            "first_active_month": ["2017-01", None],
        }
    )
    result = features(tmp_path, cards, "E05")
    months = result.filter(regex=r"^lag_")
    assert set(months.columns) == {f"lag_{lag}_count" for lag in range(14)}
    assert months.sum(axis=1).tolist() == [2, 0]
    assert result["history_count"].tolist() == [3, 0]


def test_e04_mixture_and_validation_label_independence(monkeypatch):
    import train

    class Classifier:
        def __init__(self, **_):
            pass

        def fit(self, x, y):
            assert y.tolist() == [1, 0]
            return self

        def predict_proba(self, x):
            return np.tile([0.75, 0.25], (len(x), 1))

    class Regressor:
        def __init__(self, **_):
            self.mean = 0.0

        def fit(self, x, y):
            self.mean = float(y.mean())
            return self

        def predict(self, x):
            return np.full(len(x), self.mean)

    monkeypatch.setattr(train, "LGBMClassifier", Classifier)
    monkeypatch.setattr(train, "LGBMRegressor", Regressor)
    x = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = np.array([-32.0, 4.0, 5.0])
    training, valid = np.array([0, 1]), np.array([2])

    def build(target):
        output = train.fit_fold(x, target, training, valid, "E04")
        return output.valid, output.query

    prediction, test_prediction = build(y)
    np.testing.assert_array_equal(prediction, [-5.0])
    np.testing.assert_array_equal(test_prediction, [-5.0])
    check_label_independence(build, y, np.array([-32.0, 4.0, -33.0]))
