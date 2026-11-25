"""公式学習データの分布と、各 CV run の誤差を集計する。"""

import argparse
from pathlib import Path

import polars as pl
from features import (
    COMPETITION,
    DATA,
    TARGET,
    TIME,
    WEATHER,
    join_features,
    read_buildings,
    read_meter,
    read_weather,
)
from run import splits_for

from kumite.runs import RunStore


def eda(output: Path) -> None:
    joined = join_features(
        read_meter(DATA / "train.csv"),
        read_buildings(DATA / "building_metadata.csv"),
        read_weather(DATA / "weather_train.csv"),
    )
    output.mkdir(exist_ok=True)
    counts = joined.select(
        pl.len().alias("rows"), pl.col(TIME).min().alias("start"), pl.col(TIME).max().alias("end")
    )
    counts.write_csv(output / "coverage.csv")
    statistics = [
        pl.len().alias("rows"),
        (pl.col(TARGET) == 0).mean().alias("zero_rate"),
        pl.col(TARGET).mean().alias("mean_reading"),
    ]
    for name in ("building_id", "meter", "site_id"):
        joined.group_by(name).agg(*statistics).sort(name).write_csv(output / f"{name}.csv")
    joined.with_columns(hour=pl.col(TIME).dt.hour()).group_by("meter", "hour").agg(
        *statistics
    ).sort("meter", "hour").write_csv(output / "hour.csv")
    joined.group_by("site_id").agg(
        pl.col(name).is_null().mean().alias(name) for name in WEATHER
    ).sort("site_id").write_csv(output / "weather_missing.csv")
    print(counts)


def analyze(run_id: str, output: Path) -> None:
    store = RunStore(COMPETITION)
    validation = pl.read_parquet(store.artifact_path(run_id, "validation.parquet"))
    _, windows = splits_for(validation.get_column(TIME))
    fold = pl.lit(None, dtype=pl.Int8)
    for index, (start, stop) in enumerate(windows):
        fold = (
            pl.when(pl.col(TIME).is_between(start, stop, closed="left")).then(index).otherwise(fold)
        )
    output.mkdir(exist_ok=True)
    for name in ("fold", "site_id", "meter"):
        validation.with_columns(fold.alias("fold")).group_by(name).agg(
            pl.len().alias("rows"),
            pl.col("log_square_error").mean().sqrt().alias("rmsle"),
            (pl.col("predicted").log1p() - pl.col(TARGET).log1p()).mean().alias("mean_log_bias"),
        ).sort(name).write_csv(output / f"{name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = COMPETITION / "outputs"
    if args.run_id:
        analyze(args.run_id, root / f"analysis-{args.run_id}")
    else:
        eda(root / "eda")
