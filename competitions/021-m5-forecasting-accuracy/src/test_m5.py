from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dataset import HORIZON, Sales, submission, validate_prediction
from metric import Hierarchy, Scorer
from model import BASE_WIDTH, SERIES_BUCKETS, Features, seasonal, training_batches, training_origins


def example() -> Sales:
    series = pd.DataFrame(
        {
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
        }
    )
    units = np.zeros((2, 58), dtype=np.int32)
    units[0, 2:] = 2
    units[0, 3] = 4
    calendar = pd.DataFrame(
        {
            "wday": np.arange(58) % 7 + 1,
            "month": 1,
            "snap_CA": 1,
            "snap_TX": 0,
            "snap_WI": 0,
            "event_name_1": None,
            "event_name_2": None,
        }
    )
    return Sales(series, units, calendar, np.array([[3.0], [np.nan]]), np.zeros(58, dtype=int))


def test_wrmsse_hand_calculation() -> None:
    data = example()
    hierarchy = Hierarchy.from_series(data.series)
    scorer = Scorer.at(data, hierarchy, 30)
    actual = data.actual(31, 58)
    changed = actual.copy()
    changed[0] += 1
    np.testing.assert_allclose(scorer.scale[scorer.weight > 0], 8 / 27)
    np.testing.assert_allclose(scorer.score(actual, changed), np.sqrt(27 / 8))
    np.testing.assert_array_equal(scorer.score(actual, actual), np.zeros(12))
    np.testing.assert_allclose(np.bincount(hierarchy.levels, weights=scorer.weight), 1)


def test_dollar_weights_and_missing_price() -> None:
    data = example()
    data.units[1, 4] = 5
    hierarchy = Hierarchy.from_series(data.series)
    with pytest.raises(ValueError, match="正の売上に価格"):
        Scorer.at(data, hierarchy, 30)
    data.prices[1, 0] = 2
    scorer = Scorer.at(data, hierarchy, 30)
    bottom = scorer.weight[hierarchy.levels == 11]
    np.testing.assert_allclose(bottom, np.array([58 * 3, 5 * 2]) / 184)


@pytest.mark.parametrize("group", ["base", "calendar", "price", "intermittency"])
def test_future_sales_do_not_change_features_or_metric_state(group: str) -> None:
    data = example()
    hierarchy = Hierarchy.from_series(data.series)
    original = Features(data, group)
    before = original.allocate()
    original.fill(before, 30)
    scorer = Scorer.at(data, hierarchy, 30)
    expected = seasonal(data, 30)
    data.units[:, 30:] += 100
    changed = Features(data, group)
    after = changed.allocate()
    changed.fill(after, 30)
    new_scorer = Scorer.at(data, hierarchy, 30)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(scorer.scale, new_scorer.scale)
    np.testing.assert_array_equal(scorer.weight, new_scorer.weight)
    np.testing.assert_array_equal(expected, seasonal(data, 30))


def test_training_windows_and_prediction_checks() -> None:
    origins = training_origins(1829)
    assert origins[0] == 1096
    assert all(origin + HORIZON <= 1829 for origin in origins)
    assert origins[-1] + 7 + HORIZON > 1829
    validate_prediction(np.zeros((2, 28)), 2)
    with pytest.raises(ValueError):
        validate_prediction(np.full((2, 28), np.nan), 2)
    with pytest.raises(ValueError):
        validate_prediction(-np.ones((2, 28)), 2)
    with pytest.raises(ValueError):
        validate_prediction(np.zeros((28, 2)), 2)


def test_rolling_sample_covers_all_series() -> None:
    batches = training_batches(1829, 33)
    assert tuple(day for day, _ in batches) == tuple(training_origins(1829))
    first_cycle = np.concatenate([rows for _, rows in batches[:SERIES_BUCKETS]])
    np.testing.assert_array_equal(np.sort(first_cycle), np.arange(33))
    np.testing.assert_array_equal(batches[0][1], batches[SERIES_BUCKETS][1])


def test_lags_are_relative_to_target_day_and_end_at_origin() -> None:
    data = example()
    data.calendar = pd.concat([data.calendar, data.calendar], ignore_index=True)
    data.units = np.tile(np.arange(1, 117), (2, 1))
    features = Features(data, "lags")
    matrix = features.allocate()
    features.fill(matrix, 45)
    np.testing.assert_array_equal(matrix[0, BASE_WIDTH:], np.arange(18, 3, -1))
    np.testing.assert_array_equal(matrix[27, BASE_WIDTH:], np.arange(45, 30, -1))
    data.units[:, 45:] += 1000
    modified = features.allocate()
    features.fill(modified, 45)
    np.testing.assert_array_equal(matrix, modified)


def test_submission_uses_sample_id_order(tmp_path: Path) -> None:
    data = example()
    ids = [
        "FOODS_1_002_CA_1_evaluation",
        "FOODS_1_001_CA_1_validation",
        "FOODS_1_001_CA_1_evaluation",
        "FOODS_1_002_CA_1_validation",
    ]
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ids}).to_csv(sample, index=False)
    public = np.repeat([[1.0], [2.0]], HORIZON, axis=1)
    private = np.repeat([[3.0], [4.0]], HORIZON, axis=1)
    result = submission(data, public, data, private, tmp_path)
    assert result["id"].tolist() == ids
    assert result["F28"].tolist() == [4.0, 1.0, 3.0, 2.0]
    pd.DataFrame({"id": ids[:3] + ids[:1]}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="提出 ID"):
        submission(data, public, data, private, tmp_path)
