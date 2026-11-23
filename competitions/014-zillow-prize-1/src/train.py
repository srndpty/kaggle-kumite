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
    TAX_FEATURES,
    TRANSACTIONS,
    category_types,
    check_transaction_order,
    describe,
    feature_frame,
    load_train,
    read_properties,
    time_splits,
)

from kumite.bootstrap import group_bootstrap_diff
from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import ExperimentRecord
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
    "E02-no-tax": ("税額・評価額を外した絶対誤差回帰", "regression_l1"),
    "E04-no-tax": ("税額・評価額を外した条件で直近重みを検証", "regression_l1"),
}
SPLIT_VERSION = "zillow-time-v1"


def trial_params(trial: str) -> dict[str, Any]:
    return {**SETTINGS, "trial": trial, "objective": TRIALS[trial][1]}


def feature_version(trial: str) -> str:
    return "zillow-properties-v1-no-tax" if trial.endswith("-no-tax") else "zillow-properties-v1"


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
    if trial in ("E04", "E04-no-tax"):
        days = (cutoff - train[DATE]).dt.days.to_numpy(dtype=np.float64)
        weight = np.exp2(-days / 365)
    params = {**SETTINGS, "objective": TRIALS[trial][1]}
    zip_value = train.groupby("regionidzip")["taxvaluedollarcnt"].mean() if trial == "E05" else None

    def features(rows: pd.DataFrame, month: int | None = None) -> pd.DataFrame:
        frame = feature_frame(rows, categories, month)
        if trial.endswith("-no-tax"):
            frame = frame.drop(columns=list(TAX_FEATURES))
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
        "14", "MAE", False, mae, data_hash(), SPLIT_VERSION, array_sha256(roles), 42
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
                feature_version(name),
                trial_params(name),
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


def audit_runs(
    train: pd.DataFrame,
    run_ids: list[str],
) -> dict[str, tuple[ExperimentRecord, np.ndarray]]:
    check_transaction_order(train)
    inputs = data_hash()
    splits = time_splits(train[DATE])
    roles = split_roles(len(train), splits)
    split_hash = array_sha256(roles)
    valid = (roles == 2).any(axis=0)
    expected = train[DATE].ge("2017-01-01") & train[DATE].lt("2017-09-01")
    if not np.array_equal(valid, expected.to_numpy()):
        raise ValueError("検証対象行が固定期間と一致しません")
    y = train[TARGET].to_numpy()
    runs = {}
    for run_id in run_ids:
        record = STORE.record(run_id)
        trial = record.params.get("trial")
        if (
            trial not in TRIALS
            or record.params != trial_params(trial)
            or record.model != TRIALS[trial][1]
            or record.data_version != feature_version(trial)
            or record.competition_id != "14"
            or record.metric != "MAE"
            or record.greater_is_better
            or record.seed != SETTINGS["seed"]
            or record.data_hash != inputs
            or record.split_version != SPLIT_VERSION
            or record.split_hash != split_hash
        ):
            raise ValueError(f"{run_id} の入力データ・分割・学習設定が一致しません")
        saved_roles = STORE.load_array(run_id, "split_roles.npy")
        if array_sha256(saved_roles) != split_hash:
            raise ValueError(f"{run_id} の分割成果物が一致しません")
        oof = STORE.load_array(run_id, "oof.npy")
        if (
            oof.shape != y.shape
            or not np.isfinite(oof[valid]).all()
            or not np.isnan(oof[~valid]).all()
        ):
            raise ValueError(f"{run_id} の OOF 対象行が一致しません")
        fold_scores = [mae(y[indices], oof[indices]) for _, indices in splits]
        if mae(y[valid], oof[valid]) != record.oof_score or fold_scores != record.fold_scores:
            raise ValueError(f"{run_id} の保存済み MAE が OOF の再計算と一致しません")
        runs[run_id] = (record, oof)
    if len({record.data_version for record, _ in runs.values()}) != 1:
        raise ValueError("同じ特徴量系列の Zillow CV run を指定してください")
    return runs


def selection_payload(records: list[ExperimentRecord]) -> dict[str, Any]:
    ordered = sorted(
        records, key=lambda r: (r.oof_score, list(TRIALS).index(r.params["trial"]), r.run_id)
    )
    best = ordered[0]
    return {
        "selected": best.run_id,
        "data_hash": best.data_hash,
        "split_version": best.split_version,
        "split_hash": best.split_hash,
        "candidates": {
            r.run_id: {
                "oof_score": r.oof_score,
                "oof_sha256": r.artifacts["oof.npy"],
                "split_roles_sha256": r.artifacts["split_roles.npy"],
            }
            for r in ordered
        },
    }


def select(train: pd.DataFrame, run_ids: list[str], output: Path) -> None:
    audited = audit_runs(train, run_ids)
    payload = selection_payload([record for record, _ in audited.values()])
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(payload)


def verify_selection(train: pd.DataFrame, contents: bytes) -> ExperimentRecord:
    saved = json.loads(contents)
    if (
        not isinstance(saved, dict)
        or not isinstance(saved.get("candidates"), dict)
        or not saved["candidates"]
    ):
        raise ValueError("選定記録の候補がありません")
    audited = audit_runs(train, list(saved["candidates"]))
    expected = selection_payload([record for record, _ in audited.values()])
    if saved != expected:
        raise ValueError("選定記録が CV 成果物と一致しません")
    return audited[expected["selected"]][0]


def final(train: pd.DataFrame, run_id: str, source: str | None, selection: Path | None) -> None:
    contents = selection.read_bytes() if selection is not None else None
    if contents is not None:
        record = verify_selection(train, contents)
    elif source is not None:
        record = audit_runs(train, [source])[source][0]
    else:
        raise ValueError("選定記録または CV run を指定してください")
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
            print(f"{year} 年の物件 ID 未結合: {(~ids.isin(properties.index)).sum()} 件")
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
            if contents is not None:
                (out / "selection.json").write_bytes(contents)

        final_record = replace(
            record,
            run_id=run_id,
            hypothesis="CV run を再学習して提出を生成",
            changed_one_thing="年別締切で再学習",
            code_version=code_version(REPO_ROOT),
            runtime_sec=time.perf_counter() - start,
            peak_memory_mb=peak_memory_mb(),
            inputs=STORE.input_hashes(record.run_id, "oof.npy", "split_roles.npy"),
            artifacts={},
            late_score=None,
        )
        print(STORE.publish(reservation, final_record, write))


def analyze(train: pd.DataFrame, base: str, other: str) -> None:
    audited = audit_runs(train, [base, other])
    base_pred, other_pred = audited[base][1], audited[other][1]
    valid = np.isfinite(base_pred)
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
    if args.command == "cv" and args.run_id and len(args.trials) != 1:
        parser.error("--run-id は単一実験を指定する場合だけ使用できます")
    train = load_train()
    if args.command == "select":
        select(train, args.run_ids, args.output)
    elif args.command == "eda":
        describe(train)
    elif args.command == "cv":
        cv(train, args.trials, args.run_id)
    elif args.command == "analyze":
        analyze(train, args.base, args.other)
    else:
        selection = None if args.source else args.selection or DEFAULT_SELECTION
        final(train, args.run_id, args.source, selection)


if __name__ == "__main__":
    main()
