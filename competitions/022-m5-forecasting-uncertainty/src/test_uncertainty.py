from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from metric import scale_at, score
from model import EXPERIMENTS, QUANTILE_NAMES, QUANTILES, forecast
from submission import series_ids, write_submission

from kumite.m5 import HORIZON, Hierarchy, Sales, revenue_weights


def example() -> Sales:
    series = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "dept_id": ["D", "D"],
            "cat_id": ["C", "C"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
        }
    )
    return Sales(
        series,
        np.zeros((2, 430), dtype=np.int32),
        pd.DataFrame(),
        np.array([[1.0], [np.nan]]),
        np.zeros(430, dtype=int),
    )


def test_wspl_hand_calculation() -> None:
    data = example()
    data.units[0, 2:5] = [2, 5, 1]
    hierarchy = Hierarchy.from_series(data.series)
    weight = revenue_weights(data, hierarchy, 28)
    scale = scale_at(data, hierarchy, 28)
    np.testing.assert_allclose(scale[weight > 0], 8 / 25)
    np.testing.assert_equal(scale[weight == 0], 0)
    actual = np.full((len(weight), HORIZON), 2.0)
    prediction = np.ones((*actual.shape, len(QUANTILES)))
    np.testing.assert_allclose(
        score(actual, prediction, hierarchy, weight, scale), np.tile(QUANTILES * 25 / 8, (12, 1))
    )
    with pytest.raises(ValueError, match="尺度"):
        score(actual, prediction, hierarchy, weight, np.zeros_like(scale))


@pytest.mark.parametrize("experiment", tuple(EXPERIMENTS))
def test_future_labels_do_not_change_calibration(experiment: str) -> None:
    data = example()
    data.units[0] = np.arange(data.units.shape[1]) % 13
    modified = replace(data, units=data.units.copy())
    origin = 390
    modified.units[:, origin:] = 999
    hierarchy = Hierarchy.from_series(data.series)
    np.testing.assert_array_equal(
        forecast(data, hierarchy, origin, experiment),
        forecast(modified, hierarchy, origin, experiment),
    )
    np.testing.assert_array_equal(
        scale_at(data, hierarchy, origin), scale_at(modified, hierarchy, origin)
    )
    np.testing.assert_array_equal(
        revenue_weights(data, hierarchy, origin), revenue_weights(modified, hierarchy, origin)
    )


def test_submission_order_and_quantile_crossing(tmp_path: Path) -> None:
    hierarchy = Hierarchy.from_series(example().series)
    ids = series_ids(hierarchy)
    assert {"Total_X", "CA_X", "CA_A", "A_CA_1"} <= set(ids)

    sample = tmp_path / "sample.csv"
    keys = [
        f"{name}_{q}_{suffix}"
        for suffix in ("validation", "evaluation")
        for q in QUANTILE_NAMES
        for name in ids
    ][::-1]
    pd.DataFrame({"id": keys}).to_csv(sample, index=False)
    shape = (len(ids), HORIZON, len(QUANTILES))
    forecast_array = np.broadcast_to(QUANTILES, shape).copy()
    pair = (hierarchy, forecast_array)
    output = tmp_path / "output.csv"
    write_submission(output, sample, pair, pair)
    assert pd.read_csv(output)["id"].tolist() == keys
    forecast_array[:, :, 0] = 1
    with pytest.raises(ValueError, match="交差"):
        write_submission(output, sample, pair, pair)


def test_rounding_changes_only_bottom_level() -> None:
    data = example()
    data.units[0] = np.arange(data.units.shape[1]) % 13
    hierarchy = Hierarchy.from_series(data.series)
    baseline = forecast(data, hierarchy, 390, "E01")
    rounded = forecast(data, hierarchy, 390, "E05")
    bottom = hierarchy.levels == 11
    np.testing.assert_array_equal(rounded[bottom], np.rint(baseline[bottom]))
    np.testing.assert_array_equal(rounded[~bottom], baseline[~bottom])
    assert np.any(rounded[bottom] != baseline[bottom])
