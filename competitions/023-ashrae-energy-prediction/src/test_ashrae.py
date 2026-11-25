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
from run import CATEGORICAL, PARAMS, SPLIT_VERSION, splits_for, to_pandas

from kumite.experiments import ExperimentRecord
from kumite.gbdt import fit_lightgbm_fixed
from kumite.provenance import array_sha256
from kumite.runs import RunStore


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
        "row_id,building_id,meter,timestamp\n0,1,0,2017-01-01 00:00:00\n"
        "1,1,0,2017-01-01 01:00:00\n",
        encoding="utf-8",
    )
    sample.write_text("row_id,meter_reading\n0,0\n1,0\n", encoding="utf-8")
    monkeypatch.setattr(final, "TEST_FILES", [test, tmp_path / "weather.csv", sample])
    monkeypatch.setattr(final, "predict", lambda *args: np.zeros(args[1].height))
    monkeypatch.setattr(final, "CHUNK_ROWS", 1)
    output = tmp_path / "submission.csv"
    model = final.lgb.Booster(model_str=_tiny_model())
    write_submission(model, pl.DataFrame(), "E00", None, output)
    assert pl.read_csv(output).get_column("row_id").to_list() == [0, 1]
    sample.write_text("row_id,meter_reading\n1,0\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID 順"):
        write_submission(model, pl.DataFrame(), "E00", None, output)

    sample.write_text("row_id,meter_reading\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_submission(model, pl.DataFrame(), "E00", None, output)
    sample.write_text("row_id,meter_reading\n0,0\n0,0\n", encoding="utf-8")
    test.write_text(
        "row_id,building_id,meter,timestamp\n0,1,0,2017-01-01 00:00:00\n"
        "0,1,0,2017-01-01 01:00:00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="連続"):
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


@pytest.mark.parametrize(
    ("value", "outside", "tamper", "error"),
    [
        (1.0, np.nan, False, None),
        (1.0, np.nan, True, "ハッシュ"),
        (1.0, 0.0, False, "検証窓外"),
        (np.nan, np.nan, False, "検証窓内"),
        (2.0, np.nan, False, "再計算"),
    ],
)
def test_candidate_oof_validation(
    tmp_path: Path, value: float, outside: float, tamper: bool, error: str | None
) -> None:
    store = RunStore(tmp_path)
    roles = np.array([[1, 2]], dtype=np.int8)
    target = np.array([1.0, 1.0])
    record = ExperimentRecord(
        competition_id="ashrae-energy-prediction",
        run_id="E01",
        metric="rmsle",
        greater_is_better=False,
        data_version="E01",
        data_hash="train",
        split_version=SPLIT_VERSION,
        split_hash=array_sha256(roles),
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="E01",
        params=PARAMS,
        fold_scores=[0.0],
        oof_score=0.0,
        code_version="test",
    )

    def write(path: Path) -> None:
        np.save(path / "split.npy", roles)
        np.save(path / "oof.npy", np.array([outside, value]))

    with store.reserve("E01") as reservation:
        published = store.publish(reservation, record, write)
    if tamper:
        np.save(published / "oof.npy", np.array([np.nan, 2.0]))
    if error:
        with pytest.raises(ValueError, match=error):
            final.select_candidate(store, ["E01"], "train", roles, target)
    else:
        assert final.select_candidate(store, ["E01"], "train", roles, target).run_id == "E01"


def test_temperature_lag_crosses_files_without_ambiguous_keys(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text(
        "site_id,timestamp,air_temperature\n0,2016-12-31 00:00:00,11\n", encoding="utf-8"
    )
    test.write_text(
        "site_id,timestamp,air_temperature\n0,2017-01-01 00:00:00,45\n", encoding="utf-8"
    )
    lagged = weather_for(test, "E04", history=train)
    assert lagged is not None
    assert lagged.get_column(TEMPERATURE_LAG).to_list() == [11.0]

    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        weather_for(test, "E04", history=test)


def test_unknown_building_is_rejected() -> None:
    meters = pl.DataFrame({"building_id": [2]})
    buildings = pl.DataFrame({"building_id": [1], "site_id": [0]})
    with pytest.raises(ValueError, match="building_id"):
        join_features(meters, buildings)
