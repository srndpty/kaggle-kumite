from datetime import datetime, timedelta
from pathlib import Path

import final
import numpy as np
import polars as pl
import pytest
from features import (
    TARGET,
    TEMPERATURE_LAG,
    TIME,
    WEATHER,
    estimate_offsets,
    join_features,
    model_frame,
    rmsle,
    weather_for,
)
from final import predict, write_submission
from run import CATEGORICAL, PARAMS, splits_for, to_pandas

from kumite.gbdt import fit_lightgbm_fixed


def test_join_and_missing_flags() -> None:
    when = datetime(2016, 1, 1)
    meters = pl.DataFrame(
        {"building_id": [1, 1], "meter": [0, 1], TIME: [when, when + timedelta(hours=1)]}
    )
    buildings = pl.DataFrame(
        {
            "building_id": [1],
            "site_id": [0],
            "primary_use": [1],
            "square_feet": [100],
            "year_built": [1990],
            "floor_count": [1],
        }
    )
    weather = pl.DataFrame({"site_id": [0], TIME: [when], **{name: [10.0] for name in WEATHER}})
    joined = join_features(meters, buildings, weather)
    assert joined.height == 2
    assert model_frame(joined, "E02").get_column("air_temperature_missing").to_list() == [0, 1]
    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        join_features(meters, buildings, pl.concat([weather, weather]))
    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        join_features(meters, pl.concat([buildings, buildings]))


def test_validation_window_and_rmsle() -> None:
    first = datetime(2016, 1, 1)
    dates = pl.Series([first + timedelta(days=i) for i in range(100)])
    splits, _ = splits_for(dates)
    assert [len(train) for train, _ in splits] == [16, 44, 72]
    assert all(len(valid) == 28 and train.max() < valid.min() for train, valid in splits)
    assert rmsle(np.array([0.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.log(2))


def test_offsets_use_only_training_weather() -> None:
    now = datetime(2016, 1, 1)
    frame = pl.DataFrame(
        {
            TIME: [now + timedelta(hours=i) for i in (12, 16, 36, 40)],
            "site_id": [0] * 4,
            "air_temperature": [8.0, 9.0, 100.0, 0.0],
        }
    )
    assert estimate_offsets(frame, now, now + timedelta(days=1)) == {0: -1}


def test_chunk_predictions_match() -> None:
    when = datetime(2016, 1, 1)
    meters = pl.DataFrame(
        {
            "building_id": [1] * 48,
            "meter": [0] * 48,
            TIME: [when + timedelta(hours=i) for i in range(48)],
            TARGET: np.arange(48),
        }
    )
    buildings = pl.DataFrame(
        {
            "building_id": [1],
            "site_id": [0],
            "primary_use": [1],
            "square_feet": [100],
            "year_built": [1990],
            "floor_count": [1],
        }
    )
    features = to_pandas(model_frame(join_features(meters, buildings), "E00"))
    params = {**PARAMS, "num_boost_round": 5, "num_threads": 1, "min_data_in_leaf": 2}
    model = fit_lightgbm_fixed(features, np.log1p(np.arange(48)), params, CATEGORICAL)
    whole = predict(model, meters, buildings, None, "E00", None)
    chunked = np.concatenate(
        [predict(model, batch, buildings, None, "E00", None) for batch in meters.iter_slices(7)]
    )
    np.testing.assert_array_equal(chunked, whole)


def test_submission_checks_sample_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test = tmp_path / "test.csv"
    sample = tmp_path / "sample_submission.csv"
    test.write_text(
        "row_id,building_id,meter,timestamp\n1,1,0,2017-01-01 00:00:00\n"
        "2,1,0,2017-01-01 01:00:00\n",
        encoding="utf-8",
    )
    sample.write_text("row_id,meter_reading\n1,0\n2,0\n", encoding="utf-8")
    monkeypatch.setattr(final, "TEST_FILES", [test, tmp_path / "weather.csv", sample])
    monkeypatch.setattr(final, "predict", lambda *args: np.zeros(args[1].height))
    monkeypatch.setattr(final, "CHUNK_ROWS", 1)
    output = tmp_path / "submission.csv"
    model = final.lgb.Booster(model_str=_tiny_model())
    write_submission(model, pl.DataFrame(), "E00", None, output)
    assert pl.read_csv(output).get_column("row_id").to_list() == [1, 2]
    sample.write_text("row_id,meter_reading\n2,0\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID 順"):
        write_submission(model, pl.DataFrame(), "E00", None, output)


def _tiny_model() -> str:
    frame = to_pandas(pl.DataFrame({"x": [0, 1, 2]}))
    params = {**PARAMS, "num_boost_round": 1, "num_threads": 1}
    return fit_lightgbm_fixed(frame, np.array([0.0, 1.0, 2.0]), params, ()).model_to_string()


def test_temperature_lag_uses_exact_site_and_hour(tmp_path: Path) -> None:
    when = datetime(2016, 1, 1)
    weather = pl.DataFrame(
        {
            "site_id": [0, 1, 0, 1, 0],
            TIME: [
                when,
                when,
                when + timedelta(days=1),
                when + timedelta(days=1),
                when + timedelta(days=1, hours=1),
            ],
            "air_temperature": [11.0, 22.0, 45.0, 99.0, 100.0],
        }
    )
    path = tmp_path / "weather.csv"
    weather.write_csv(path)
    lagged = weather_for(path, "E04")
    assert lagged is not None
    assert lagged.get_column(TEMPERATURE_LAG).to_list() == [None, None, 11.0, 22.0, None]
