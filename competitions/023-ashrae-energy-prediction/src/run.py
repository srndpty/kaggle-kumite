"""固定した将来期間で ASHRAE の特徴量を比較する。"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
from features import (
    CATEGORICAL,
    COMPETITION,
    DATA,
    TARGET,
    TIME,
    estimate_offsets,
    join_features,
    model_frame,
    read_buildings,
    read_meter,
    restore,
    rmsle,
    validation_windows,
    weather_for,
)

from kumite.crossfit import FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import fit_lightgbm_fixed
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

SPLIT_VERSION = "expanding-28d-v1"
SEED = 42
PARAMS = {
    "objective": "regression",
    "num_boost_round": 500,
    "early_stopping": None,
    "seed": SEED,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_bin": 63,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 16,
}
METHODS = {
    "E00": "建物・メーター・カレンダー",
    "E01": "冷暖房メーターでは気象値の追加で改善が大きい",
    "E02": "欠測率の高いサイトでは気象欠測フラグが効く",
    "E03": "気温ピークから推定した時刻補正は、時差が大きいサイトで効く",
    "E04": "E02 に24時間前の気温を加えると、建物の熱慣性を捉えやすい",
}
TRAIN_FILES = [DATA / f"{name}.csv" for name in ("train", "building_metadata", "weather_train")]


def to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_numpy(), columns=frame.columns, copy=False)


def splits_for(dates: pl.Series) -> tuple[list[Split], list[tuple[datetime, datetime]]]:
    last = dates.max()
    if not isinstance(last, datetime):
        raise ValueError("日時がありません")
    windows = validation_windows(last)
    values = dates.to_numpy().astype("datetime64[us]")
    splits = [
        (
            np.flatnonzero(values < np.datetime64(start)),
            np.flatnonzero((values >= np.datetime64(start)) & (values < np.datetime64(stop))),
        )
        for start, stop in windows
    ]
    return splits, windows


def run(method: str, run_id: str) -> None:
    store = RunStore(COMPETITION)
    with store.reserve(run_id) as reservation:
        started = perf_counter()
        data_hash = files_sha256(TRAIN_FILES)
        meters = read_meter(TRAIN_FILES[0])
        buildings = read_buildings(TRAIN_FILES[1])
        weather = weather_for(TRAIN_FILES[2], method)
        joined = join_features(meters, buildings, weather)
        y = joined.get_column(TARGET).to_numpy()
        dates = joined.get_column(TIME)
        first = dates.min()
        assert isinstance(first, datetime)
        splits, windows = splits_for(dates)
        roles = split_roles(len(y), splits)
        inputs = store.input_hashes("E02", "oof.npy", "split.npy") if method == "E04" else {}
        if method == "E04":
            parent = store.record("E02")
            if parent.data_hash != data_hash or parent.split_hash != array_sha256(roles):
                raise ValueError("E02 のデータまたは分割が一致しません")
        fixed = to_pandas(model_frame(joined, method)) if method != "E03" else None
        offset_history: list[dict[int, int]] = []

        def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
            info: dict[str, object] = {}
            if fixed is None:
                assert weather is not None
                start = windows[len(offset_history)][0]
                offsets = estimate_offsets(weather, first, start)
                offset_history.append(offsets)
                info["offsets"] = offsets
                x_train = to_pandas(model_frame(joined[train], method, offsets))
                x_valid = to_pandas(model_frame(joined[valid], method, offsets))
            else:
                x_train = fixed.iloc[train]
                x_valid = fixed.iloc[valid]
            print(f"学習開始: {len(train):,} 行、検証 {len(valid):,} 行", flush=True)
            model = fit_lightgbm_fixed(x_train, np.log1p(y[train]), PARAMS, CATEGORICAL)
            predicted = restore(np.asarray(model.predict(x_valid), dtype=np.float64))
            print(f"RMSLE {rmsle(y[valid], predicted):.6f}", flush=True)
            return FoldOutput(predicted, np.empty(0), info)

        result = cross_fit_splits(len(y), splits, fit)
        evaluation = Evaluation(
            "ashrae-energy-prediction",
            "rmsle",
            False,
            rmsle,
            data_hash,
            SPLIT_VERSION,
            array_sha256(roles),
            SEED,
        )

        def write(path: Path) -> None:
            np.save(path / "split.npy", roles)
            boundaries = [[start.isoformat(), stop.isoformat()] for start, stop in windows]
            (path / "windows.json").write_text(json.dumps(boundaries), encoding="utf-8")
            valid = np.flatnonzero(np.isfinite(result.oof))
            rows = joined[valid].select("building_id", "site_id", "meter", TIME, TARGET)
            rows.with_columns(
                predicted=pl.Series(result.oof[valid]),
                log_square_error=pl.Series(
                    (np.log1p(y[valid].astype(np.float64)) - np.log1p(result.oof[valid])) ** 2
                ),
            ).write_parquet(path / "validation.parquet")

        publish_cv_run(
            store,
            reservation,
            evaluation,
            Trial(METHODS[method], METHODS[method], method, method, PARAMS, inputs),
            y,
            splits,
            result,
            perf_counter() - started,
            write,
            save_query=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run(args.method, args.run_id or args.method)
