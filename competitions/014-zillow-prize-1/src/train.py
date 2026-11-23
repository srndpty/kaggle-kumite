"""Zillow の EDA、時系列 CV、モデル選定と提出生成。"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dummy import MONTHS
from zillow import (
    CATEGORICAL,
    DATE,
    ID,
    PROPERTIES,
    ROOT,
    SAMPLE,
    TARGET,
    TRANSACTIONS,
    category_types,
    describe,
    feature_frame,
    load_train,
    read_properties,
    time_splits,
)

from kumite.bootstrap import group_bootstrap_diff
from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import fit_lightgbm_fixed
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

STORE = RunStore(ROOT)
DEFAULT_SELECTION = ROOT / "selection.json"
SETTINGS: dict[str, Any] = {
    "seed": 42,
    "num_threads": 8,
    "num_boost_round": 300,
    "early_stopping": None,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "deterministic": True,
    "force_col_wise": True,
}
TRIALS = {
    "median": ("学習ラベルの中央値を定数基準とする", "median"),
    "E01": ("物件属性と取引月から二乗誤差で学習", "regression"),
    "E02": ("絶対誤差で学習すると全体 MAE が下がる", "regression_l1"),
    "E03": ("学習ラベルの両端を抑えると全体 MAE が下がる", "regression"),
    "E04": ("直近の取引を重視すると後半の検証窓で改善する", "regression_l1"),
    "E05": ("郵便番号ごとの平均評価額を追加すると全体 MAE が下がる", "regression"),
}


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


def training_target(train: pd.DataFrame, trial: str) -> np.ndarray:
    target = train[TARGET].to_numpy(dtype=np.float64)
    if trial == "E03":
        lower, upper = np.quantile(target, [0.01, 0.99])
        return np.clip(target, lower, upper)
    return target


def fit_predictor(train: pd.DataFrame, trial: str, cutoff: pd.Timestamp):
    target = training_target(train, trial)
    if trial == "median":
        value = float(np.median(target))
        return lambda rows, month=None: np.full(len(rows), value, dtype=np.float64)
    categories = category_types(train)
    weight = None
    if trial == "E04":
        days = (cutoff - train[DATE]).dt.days.to_numpy(dtype=np.float64)
        weight = np.exp2(-days / 365)
    params = {**SETTINGS, "objective": TRIALS[trial][1]}
    zip_value = train.groupby("regionidzip")["taxvaluedollarcnt"].mean() if trial == "E05" else None

    def features(rows: pd.DataFrame, month: int | None = None) -> pd.DataFrame:
        frame = feature_frame(rows, categories, month)
        if zip_value is not None:
            frame["zip_mean_tax_value"] = rows["regionidzip"].map(zip_value)
        return frame

    booster = fit_lightgbm_fixed(
        features(train),
        target,
        params,
        CATEGORICAL,
        sample_weight=weight,
    )

    def predict(rows: pd.DataFrame, month: int | None = None) -> np.ndarray:
        return np.asarray(booster.predict(features(rows, month)), dtype=np.float64)

    return predict


def data_hash() -> str:
    return files_sha256([*TRANSACTIONS.values(), *PROPERTIES.values(), SAMPLE])


def cv(train: pd.DataFrame, trials: list[str], run_id: str | None) -> None:
    splits = time_splits(train[DATE])
    roles = split_roles(len(train), splits)
    expected = train[DATE].ge("2017-01-01") & train[DATE].lt("2017-09-01")
    if not np.array_equal((roles == 2).sum(axis=0), expected.to_numpy()):
        raise ValueError("検証対象の所属が一致しません")
    evaluation = Evaluation(
        "14", "MAE", False, mae, data_hash(), "zillow-time-v1", array_sha256(roles), 42
    )
    for name in trials:
        with STORE.reserve(run_id or name) as reservation:
            start = time.perf_counter()

            def fit(tr: np.ndarray, va: np.ndarray, trial: str = name) -> FoldOutput:
                cutoff = train.iloc[va][DATE].min().replace(day=1)
                predict = fit_predictor(train.iloc[tr], trial, cutoff)
                return FoldOutput(predict(train.iloc[va]), np.empty(0))

            result = cross_fit_splits(len(train), splits, fit)
            hypothesis, model = TRIALS[name]
            trial = Trial(
                hypothesis,
                hypothesis,
                model,
                "zillow-properties-v1",
                {**SETTINGS, "trial": name, "objective": model},
            )
            publish_cv_run(
                STORE,
                reservation,
                evaluation,
                trial,
                train[TARGET].to_numpy(),
                splits,
                result,
                time.perf_counter() - start,
                lambda out: np.save(out / "split_roles.npy", roles),
                save_query=False,
            )


def select(run_ids: list[str], output: Path) -> None:
    records = [STORE.record(run_id) for run_id in run_ids]
    first = records[0]
    if any(
        r.data_hash != first.data_hash
        or r.split_version != first.split_version
        or r.split_hash != first.split_hash
        or r.metric != "MAE"
        or r.params.get("trial") not in TRIALS
        for r in records
    ):
        raise ValueError("同じデータ・分割の Zillow CV run を指定してください")
    ordered = sorted(records, key=lambda r: (r.oof_score, list(TRIALS).index(r.params["trial"])))
    selected = ordered[0]
    payload = {"selected": selected.run_id, "candidates": {r.run_id: r.oof_score for r in ordered}}
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(payload)


def final(train: pd.DataFrame, run_id: str, source: str, selection: Path | None) -> None:
    record = STORE.record(source)
    if record.data_hash != data_hash() or record.params.get("trial") not in TRIALS:
        raise ValueError("入力データまたは CV run が一致しません")
    sample = pd.read_csv(SAMPLE, usecols=["ParcelId", *MONTHS])
    ids = sample["ParcelId"]
    if ids.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError("提出 ID が不正です")
    with STORE.reserve(run_id) as reservation:
        start = time.perf_counter()
        for year in (2016, 2017):
            cutoff = pd.Timestamp(year, 10, 15)
            predict = fit_predictor(
                train.loc[train[DATE].lt(cutoff)], record.params["trial"], cutoff
            )
            properties = read_properties(year)
            cols = [month for month in MONTHS if month.startswith(str(year))]
            for col in cols:
                sample[col] = np.nan
            for offset in range(0, len(sample), 200_000):
                index = sample.index[offset : offset + 200_000]
                rows = properties.reindex(ids.iloc[index].to_numpy())
                for col in cols:
                    predicted = predict(rows, int(col[-2:]))
                    if predicted.shape != (len(rows),) or not np.isfinite(predicted).all():
                        raise ValueError("提出予測に形状不一致または非有限値があります")
                    sample.loc[index, col] = predicted
            del properties
        if (
            list(sample.columns) != ["ParcelId", *MONTHS]
            or not np.isfinite(sample[list(MONTHS)]).all().all()
        ):
            raise ValueError("提出列または予測値が不正です")

        def write(out: Path) -> None:
            sample.to_csv(out / "submission.csv", index=False, float_format="%.8f")
            if selection is not None:
                (out / "selection.json").write_bytes(selection.read_bytes())

        final_record = replace(
            record,
            run_id=run_id,
            hypothesis="CV run を再学習して提出を生成",
            changed_one_thing="年別締切で再学習",
            code_version=code_version(REPO_ROOT),
            runtime_sec=time.perf_counter() - start,
            peak_memory_mb=peak_memory_mb(),
            inputs=STORE.input_hashes(source, "oof.npy", "split_roles.npy"),
            artifacts={},
            late_score=None,
        )
        print(STORE.publish(reservation, final_record, write))


def analyze(train: pd.DataFrame, base: str, other: str) -> None:
    base_pred = STORE.load_array(base, "oof.npy")
    other_pred = STORE.load_array(other, "oof.npy")
    valid = np.isfinite(base_pred)
    if not np.array_equal(valid, np.isfinite(other_pred)):
        raise ValueError("OOF 対象行が一致しません")
    rows = train.loc[valid].copy()
    actual = rows[TARGET].to_numpy()
    rows["base"] = np.abs(actual - base_pred[valid])
    rows["other"] = np.abs(actual - other_pred[valid])
    print(
        asdict(
            group_bootstrap_diff(
                rows["base"].to_numpy(),
                rows["other"].to_numpy(),
                rows[ID].to_numpy(),
                transform=lambda values: values,
            )
        )
    )
    low, high = np.quantile(actual, [0.05, 0.95])
    rows["tail"] = np.where(actual < low, "下側5%", np.where(actual > high, "上側5%", "中央90%"))
    for by in (rows[DATE].dt.to_period("M"), "fips", "tail"):
        print(rows.groupby(by, dropna=False)[["base", "other"]].agg(["size", "mean"]).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("eda")
    cv_parser = commands.add_parser("cv")
    cv_parser.add_argument("trials", nargs="+", choices=list(TRIALS))
    cv_parser.add_argument("--run-id")
    select_parser = commands.add_parser("select")
    select_parser.add_argument("run_ids", nargs="+")
    select_parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION)
    final_parser = commands.add_parser("final")
    final_parser.add_argument("run_id")
    sources = final_parser.add_mutually_exclusive_group()
    sources.add_argument("--selection", type=Path)
    sources.add_argument("--source")
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("base")
    analyze_parser.add_argument("other")
    args = parser.parse_args()
    if args.command == "select":
        select(args.run_ids, args.output)
        return
    if args.command == "cv" and args.run_id and len(args.trials) != 1:
        parser.error("--run-id は単一実験を指定する場合だけ使用できます")
    train = load_train()
    if args.command == "eda":
        describe(train)
    elif args.command == "cv":
        cv(train, args.trials, args.run_id)
    elif args.command == "analyze":
        analyze(train, args.base, args.other)
    else:
        selection = None if args.source else args.selection or DEFAULT_SELECTION
        source = (
            args.source
            if selection is None
            else json.loads(selection.read_text(encoding="utf-8"))["selected"]
        )
        final(train, args.run_id, source, selection)


if __name__ == "__main__":
    main()
