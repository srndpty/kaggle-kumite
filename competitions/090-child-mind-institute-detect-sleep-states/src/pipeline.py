"""センサーの1分特徴量とイベント候補。"""

import base64
import gzip
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.ndimage import maximum_filter1d

EVENTS = ("onset", "wakeup")
COLUMNS = ["series_id", "step", "event", "score"]


def bins(path: Path, clock: bool = False) -> pl.DataFrame:
    steps = pl.col("step").cast(pl.Int64)
    extra = [pl.col("timestamp")] if clock else []
    time_agg = [pl.col("timestamp").first()] if clock else []
    frame = (
        pl.scan_parquet(path)
        .select("series_id", steps, "anglez", "enmo", *extra)
        .sort("series_id", "step")
        .with_columns(steps.diff().over("series_id").alias("step_delta"))
        .with_columns(
            (pl.col("step_delta") != 1).fill_null(True).alias("gap"),
            (steps // 12).alias("minute"),
        )
        .with_columns(pl.col("gap").cum_sum().over("series_id").alias("segment"))
        .group_by("series_id", "segment", "minute")
        .agg(
            pl.col("step").min().alias("first"),
            pl.col("step").max().alias("last"),
            pl.col("anglez", "enmo").mean(),
            (pl.col("step_delta") == 0).any().alias("duplicate_step"),
            *time_agg,
        )
        .with_columns(((pl.col("first") + pl.col("last")) // 2).alias("step"))
        .sort("series_id", "step")
        .collect()
    )

    if frame["duplicate_step"].any():
        raise ValueError("Duplicate step within series")
    return frame.drop("duplicate_step")


def features(frame: pl.DataFrame, long_window: bool) -> np.ndarray:
    values = frame.select("anglez", "enmo").to_numpy().astype(np.float64)
    index = np.arange(len(values))
    columns = [values]
    for width in (5, 30) if long_window else (5,):
        prefix = np.vstack((np.zeros((1, 2)), values.cumsum(axis=0)))
        squared = np.vstack((np.zeros((1, 2)), (values**2).cumsum(axis=0)))
        means = []
        for lo, hi in (
            (np.maximum(0, index - width), index + 1),
            (index, np.minimum(len(index), index + width + 1)),
        ):
            count = (hi - lo)[:, None]
            mean = (prefix[hi] - prefix[lo]) / count
            variance = (squared[hi] - squared[lo]) / count - mean**2
            columns.extend((mean, np.sqrt(np.maximum(variance, 0))))
            means.append(mean)
        columns.append(means[1] - means[0])
    return np.column_stack(columns).astype(np.float32)


def matrix(frame: pl.DataFrame, long_window: bool, clock: bool = False) -> np.ndarray:
    values = np.concatenate(
        [
            features(part, long_window)
            for part in frame.partition_by("series_id", "segment", maintain_order=True)
        ]
    )
    if clock:
        timestamp = pl.col("timestamp")
        minute = (
            frame.select(
                timestamp.str.slice(11, 2).cast(pl.Int32) * 60
                + timestamp.str.slice(14, 2).cast(pl.Int32)
            )
            .to_series()
            .to_numpy()
        )
        phase = 2 * np.pi * minute / 1440
        values = np.column_stack((values, np.sin(phase), np.cos(phase))).astype(np.float32)
    return values


def candidates(frame: pl.DataFrame, minutes: int) -> pl.DataFrame:
    rows: list[tuple[str, int, str, float]] = []
    for (series_id,), series in frame.group_by("series_id", maintain_order=True):
        steps = series["step"].to_numpy()
        for event in EVENTS:
            peaks = []
            offset = 0
            for segment in series.partition_by("segment", maintain_order=True):
                scores = segment[event].to_numpy()
                local = np.flatnonzero(scores == maximum_filter1d(scores, size=3, mode="nearest"))
                peaks.append(local + offset)
                offset += len(scores)
            indices = np.concatenate(peaks)
            scores = series[event].to_numpy()
            order = indices[np.lexsort((steps[indices], -scores[indices]))]
            suppressed = np.zeros(len(series), dtype=bool)
            for i in order:
                if suppressed[i]:
                    continue
                step = int(steps[i])
                rows.append((series_id, step, event, float(scores[i])))
                lo, hi = np.searchsorted(steps, [step - minutes * 12, step + minutes * 12 + 1])
                suppressed[lo:hi] = True
    return pl.DataFrame(
        rows,
        schema={"series_id": pl.String, "step": pl.Int64, "event": pl.String, "score": pl.Float64},
        orient="row",
    ).sort("series_id", "step", "event")


def submission(events: pl.DataFrame, path: Path) -> pl.DataFrame:
    if (
        events.columns != COLUMNS
        or not events["step"].dtype.is_integer()
        or not events["event"].is_in(EVENTS).all()
        or not events["score"].is_between(0, 1).all()
        or events.null_count().sum_horizontal().item() > 0
    ):
        raise ValueError("Invalid event predictions")
    steps = pl.scan_parquet(path).select("series_id", pl.col("step").cast(pl.Int64))
    if events.lazy().join(steps, on=["series_id", "step"], how="anti").collect().height:
        raise ValueError("Unknown test series or step")
    return events.with_row_index("row_id")


def limit_candidates(events: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """系列の先頭から末尾までのstep幅を日換算し、各イベントの候補上限を決める。"""
    days = frame.group_by("series_id").agg(
        (((pl.col("step").max() - pl.col("step").min()) // 17280 + 1) * 8).alias("budget")
    )
    return (
        events.sort(["series_id", "event", "score", "step"], descending=[False, False, True, False])
        .with_columns(pl.col("step").cum_count().over("series_id", "event").alias("rank"))
        .join(days, on="series_id")
        .filter(pl.col("rank") <= pl.col("budget"))
        .select(COLUMNS)
        .sort("series_id", "step", "event")
    )


def read_model_bundle(path: Path, expected_sha256: str) -> dict[str, list[str]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Model bundle SHA-256 mismatch")
    return json.loads(payload)


def predict(
    path: Path, models: dict[str, list[str]], *, long_window: bool, clock: bool, suppression: int
) -> pl.DataFrame:
    frame = bins(path, clock)
    x = matrix(frame, long_window, clock)
    probabilities = frame.select("series_id", "segment", "step")
    for event in EVENTS:
        scores = []
        for payload in models[event]:
            source = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
            model = lgb.Booster(model_str=source)
            scores.append(model.predict(x, num_threads=4))
        mean = np.mean(scores, axis=0)
        if not (np.isfinite(mean).all() and ((mean >= 0) & (mean <= 1)).all()):
            raise ValueError("Invalid test probabilities")
        probabilities = probabilities.with_columns(pl.Series(event, mean))
    events = limit_candidates(candidates(probabilities, suppression), probabilities)
    return submission(events, path)
