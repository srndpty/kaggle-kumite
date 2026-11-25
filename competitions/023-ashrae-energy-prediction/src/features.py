"""公式の建物・気象データを時間単位で結合する。"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from kumite.checks import validate_regression

COMPETITION = Path(__file__).resolve().parents[1]
DATA = COMPETITION / "data"
TARGET = "meter_reading"
TIME = "timestamp"
WEATHER = (
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
)
BUILDING = ("building_id", "site_id", "primary_use", "square_feet", "year_built", "floor_count")
CALENDAR = ("hour", "weekday", "month")
CATEGORICAL = ("building_id", "site_id", "primary_use", "meter")
CHUNK_ROWS = 500_000
TEMPERATURE_LAG = "air_temperature_lag_24h"


def read_meter(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides={"building_id": pl.Int32, "meter": pl.Int8, TARGET: pl.Float32},
    )


def read_buildings(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        schema_overrides={
            "building_id": pl.Int32,
            "floor_count": pl.String,
            "square_feet": pl.String,
        },
    )
    categories = sorted(frame.get_column("primary_use").unique().to_list())
    return frame.with_columns(
        pl.col("primary_use").replace_strict(categories, list(range(len(categories)))),
        pl.col("floor_count", "square_feet").str.strip_chars().cast(pl.Int32),
    )


def read_weather(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path, try_parse_dates=True, schema_overrides=dict.fromkeys(WEATHER, pl.Float32)
    )


def weather_for(path: Path, method: str) -> pl.DataFrame | None:
    if method == "E00":
        return None
    frame = read_weather(path)
    if method == "E04":
        previous_day = frame.select(
            "site_id",
            (pl.col(TIME) + pl.duration(days=1)).alias(TIME),
            pl.col("air_temperature").alias(TEMPERATURE_LAG),
        )
        frame = frame.join(
            previous_day, on=["site_id", TIME], how="left", validate="1:1", maintain_order="left"
        )
    return frame


def join_features(
    meters: pl.DataFrame, buildings: pl.DataFrame, weather: pl.DataFrame | None = None
) -> pl.DataFrame:
    joined = meters.join(
        buildings, on="building_id", how="left", validate="m:1", maintain_order="left"
    )
    if joined.get_column("site_id").null_count():
        raise ValueError("建物メタデータにない building_id があります")
    if weather is not None:
        joined = joined.join(
            weather, on=["site_id", TIME], how="left", validate="m:1", maintain_order="left"
        )
    if joined.height != meters.height:
        raise ValueError("結合で行数が変化しました")
    return joined


def estimate_offsets(weather: pl.DataFrame, start: datetime, stop: datetime) -> dict[int, int]:
    """学習期間内の観測から気温ピークが15時に来る整数時差を求める。"""
    daily = (
        weather.filter(pl.col(TIME).is_between(start, stop, closed="left"))
        .drop_nulls("air_temperature")
        .with_columns(day=pl.col(TIME).dt.date(), hour=pl.col(TIME).dt.hour())
        .with_columns(
            centered=pl.col("air_temperature")
            - pl.col("air_temperature").mean().over("site_id", "day")
        )
    )
    peaks = (
        daily.group_by("site_id", "hour")
        .agg(pl.col("centered").mean())
        .sort(["site_id", "centered", "hour"], descending=[False, True, False])
        .unique("site_id", keep="first")
    )
    return {
        int(site): (15 - int(hour) + 12) % 24 - 12
        for site, hour in peaks.select("site_id", "hour").iter_rows()
    }


def model_frame(
    joined: pl.DataFrame, method: str, offsets: dict[int, int] | None = None
) -> pl.DataFrame:
    timestamp = pl.col(TIME)
    if offsets is not None:
        timestamp = timestamp + pl.duration(
            hours=pl.col("site_id").replace_strict(offsets, return_dtype=pl.Int32)
        )
    frame = joined.with_columns(
        hour=timestamp.dt.hour(), weekday=timestamp.dt.weekday(), month=timestamp.dt.month()
    )
    columns = [*BUILDING, "meter", *CALENDAR]
    if method != "E00":
        columns.extend(WEATHER)
    if method in ("E02", "E04"):
        flags = [f"{name}_missing" for name in WEATHER]
        frame = frame.with_columns(
            pl.col(name).is_null().alias(flag) for name, flag in zip(WEATHER, flags, strict=True)
        )
        columns.extend(flags)
    if method == "E04":
        columns.append(TEMPERATURE_LAG)
    return frame.select(pl.col(name).cast(pl.Float32) for name in columns)


def validation_windows(last: datetime) -> list[tuple[datetime, datetime]]:
    stop = last.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return [(stop - timedelta(days=28 * k), stop - timedelta(days=28 * (k - 1))) for k in (3, 2, 1)]


def rmsle(y: np.ndarray, predicted: np.ndarray) -> float:
    validate_regression(predicted, len(y), lower=0)
    error = np.log1p(predicted) - np.log1p(y.astype(np.float64))
    return float(np.sqrt(np.mean(np.square(error))))


def restore(predicted: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(predicted), 0)
