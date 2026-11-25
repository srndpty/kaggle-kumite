"""CV で選定した構成を全学習データで学習し、提出ファイルを出力する。"""

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta
from math import isclose, isfinite
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import polars as pl
from features import (
    CATEGORICAL,
    CHUNK_ROWS,
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
    weather_for,
)
from run import METHODS, PARAMS, SPLIT_VERSION, TRAIN_FILES, splits_for, to_pandas

from kumite.checks import validate_regression
from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.gbdt import fit_lightgbm_fixed
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.runs import RunStore

TEST_FILES = [DATA / f"{name}.csv" for name in ("test", "weather_test", "sample_submission")]
RMSLE_TOLERANCE = 1e-5
COMPLEXITY = {"E00": 0, "E01": 1, "E02": 2, "E03": 2, "E04": 3}


def predict(
    model: lgb.Booster,
    batch: pl.DataFrame,
    buildings: pl.DataFrame,
    weather: pl.DataFrame | None,
    method: str,
    offsets: dict[int, int] | None,
) -> np.ndarray:
    frame = to_pandas(model_frame(join_features(batch, buildings, weather), method, offsets))
    predicted = restore(np.asarray(model.predict(frame), dtype=np.float64))
    validate_regression(predicted, batch.height, lower=0)
    return predicted


def write_submission(
    model: lgb.Booster,
    buildings: pl.DataFrame,
    method: str,
    offsets: dict[int, int] | None,
    path: Path,
) -> None:
    test, weather_path, sample_path = TEST_FILES
    weather = weather_for(weather_path, method, history=TRAIN_FILES[2])
    batches = pl.scan_csv(
        test,
        try_parse_dates=True,
        schema_overrides={"building_id": pl.Int32, "meter": pl.Int8},
    ).collect_batches(chunk_size=CHUNK_ROWS)
    with sample_path.open(encoding="utf-8-sig", newline="") as source, path.open("wb") as output:
        sample = csv.reader(source)
        if next(sample) != ["row_id", TARGET]:
            raise ValueError("公式サンプルのヘッダーが想定外です")
        written = 0
        for index, batch in enumerate(batches):
            ids = batch.get_column("row_id").to_numpy()
            expected = np.fromiter(
                (int(row[0]) for row in sample), dtype=np.int64, count=batch.height
            )
            if not np.array_equal(ids, expected):
                raise ValueError("test.csv の ID 順が公式サンプルと一致しません")
            if not np.array_equal(ids, np.arange(written, written + batch.height)):
                raise ValueError("提出 ID は 0 から連続している必要があります")
            written += batch.height
            predictions = predict(model, batch, buildings, weather, method, offsets)
            pl.DataFrame({"row_id": ids, TARGET: predictions}).write_csv(
                output, include_header=index == 0
            )
        if not written or next(sample, None) is not None:
            raise ValueError("全テスト行の予測が作成されていません")


def choose_candidate(
    candidates: list[ExperimentRecord], rmsle_tolerance: float
) -> ExperimentRecord:
    if not isfinite(rmsle_tolerance) or rmsle_tolerance < 0:
        raise ValueError("RMSLE の許容差には有限の非負値を指定してください")
    best = min(row.oof_score for row in candidates)
    equivalent = [row for row in candidates if row.oof_score <= best + rmsle_tolerance]
    return min(equivalent, key=lambda row: (COMPLEXITY[row.model], row.run_id))


def select_candidate(
    store: RunStore,
    candidate_ids: list[str],
    data_hash: str,
    roles: np.ndarray,
    target: np.ndarray,
    *,
    rmsle_tolerance: float = RMSLE_TOLERANCE,
) -> ExperimentRecord:
    split_hash = array_sha256(roles)
    valid = np.any(roles == 2, axis=0)
    candidates = [store.record(run_id) for run_id in candidate_ids]
    for candidate in candidates:
        run_id = candidate.run_id
        if (
            candidate.model not in METHODS
            or candidate.params != PARAMS
            or candidate.split_version != SPLIT_VERSION
            or candidate.split_hash != split_hash
            or candidate.data_hash != data_hash
            or candidate.competition_id != "ashrae-energy-prediction"
            or candidate.metric != "rmsle"
            or candidate.greater_is_better
            or not candidate.fold_scores
        ):
            raise ValueError(f"選定対象の CV 条件が一致しません: {run_id}")
        saved_roles = store.load_array(run_id, "split.npy")
        if not np.array_equal(saved_roles, roles) or saved_roles.dtype != roles.dtype:
            raise ValueError(f"保存済みの分割が一致しません: {run_id}")
        oof = store.load_array(run_id, "oof.npy")
        if oof.shape != target.shape or not np.issubdtype(oof.dtype, np.floating):
            raise ValueError(f"OOF の形状または型が不正です: {run_id}")
        if not np.isnan(oof[~valid]).all():
            raise ValueError(f"検証窓外の OOF は NaN が必要です: {run_id}")
        try:
            score = rmsle(target[valid], oof[valid])
        except ValueError as error:
            raise ValueError(f"検証窓内の OOF が不正です: {run_id}") from error
        if not isclose(score, candidate.oof_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"OOF の再計算スコアがログと一致しません: {run_id}")
    return choose_candidate(candidates, rmsle_tolerance)


def final(run_id: str, candidate_ids: list[str], *, rmsle_tolerance: float) -> None:
    store = RunStore(COMPETITION)
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        data_hash = files_sha256(TRAIN_FILES)
        meters = read_meter(TRAIN_FILES[0])
        splits, _ = splits_for(meters.get_column(TIME))
        roles = split_roles(meters.height, splits)
        selected = select_candidate(
            store,
            candidate_ids,
            data_hash,
            roles,
            meters[TARGET].to_numpy(),
            rmsle_tolerance=rmsle_tolerance,
        )
        inputs = store.input_hashes(selected.run_id, "oof.npy", "split.npy", "validation.parquet")
        print(f"選定: {selected.run_id} OOF RMSLE {selected.oof_score:.9f}", flush=True)
        method = selected.model
        buildings = read_buildings(TRAIN_FILES[1])
        weather = weather_for(TRAIN_FILES[2], method)
        offsets = None
        if method == "E03":
            assert weather is not None
            first, last = meters.get_column(TIME).min(), meters.get_column(TIME).max()
            assert isinstance(first, datetime) and isinstance(last, datetime)
            offsets = estimate_offsets(weather, first, last + timedelta(hours=1))
        joined = join_features(meters, buildings, weather)
        x = to_pandas(model_frame(joined, method, offsets))
        y = np.log1p(joined.get_column(TARGET).to_numpy())
        model = fit_lightgbm_fixed(x, y, PARAMS, CATEGORICAL)
        del x, y, joined, meters
        test_hash = files_sha256(TEST_FILES)

        def write(path: Path) -> None:
            model.save_model(str(path / "model.txt"))
            write_submission(model, buildings, method, offsets, path / "submission.csv")
            meta = {
                "candidates": candidate_ids,
                "selected": selected.run_id,
                "rmsle_tolerance": rmsle_tolerance,
                "method": method,
                "offsets": offsets,
                "test_hash": test_hash,
            }
            (path / "selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        record = replace(
            selected,
            run_id=run_id,
            hypothesis="選定済み構成を全学習行で再学習する",
            changed_one_thing="学習に全学習行を使う",
            fold_scores=[],
            late_score=None,
            runtime_sec=perf_counter() - start,
            peak_memory_mb=None,
            code_version=code_version(COMPETITION.parents[1]),
            inputs=inputs,
            conclusion="oof_score は選定した CV run の参照値",
        )
        print(store.publish(reservation, record, write))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--rmsle-tolerance", type=float, default=RMSLE_TOLERANCE)
    args = parser.parse_args()
    final(args.run_id, args.candidates, rmsle_tolerance=args.rmsle_tolerance)
