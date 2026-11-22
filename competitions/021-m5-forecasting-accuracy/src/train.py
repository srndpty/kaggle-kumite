"""M5 の CV・選定・最終予測を実行し、再現に必要な証跡を保存する。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import (
    COMMON_FILES,
    DATA,
    EVALUATION_FILE,
    HORIZON,
    ORIGINS,
    PRIVATE_ORIGIN,
    PUBLIC_ORIGIN,
    ROOT,
    VALIDATION_FILE,
    Sales,
    load_sales,
    submission,
)
from metric import Hierarchy, Scorer
from model import (
    FIRST_TRAIN_ORIGIN,
    PARAMS,
    ROUNDS,
    SEED,
    SERIES_BUCKETS,
    STRIDE,
    predict,
    seasonal,
)

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

STORE = RunStore(ROOT)
SPLIT_VERSION = "m5-28d-v1"
SELECTION = ROOT / "selection.json"
EXPERIMENTS = {
    "E00": ("seasonal", "過去4週の同曜日平均", "季節ナイーブ"),
    "E01": ("base", "複数系列の共通学習で階層全体の誤差を下げる", "LightGBM 共通回帰"),
    "E02": ("calendar", "暦要因の大きい日の誤差を下げる", "イベント・州別 SNAP を追加"),
    "E03": ("price", "価格変更のある系列の誤差を下げる", "予定価格・過去平均との比を追加"),
    "E04": (
        "intermittency",
        "間欠需要系列の誤差を下げる",
        "最終非ゼロからの日数・非ゼロ日率を追加",
    ),
    "E05": ("lags", "過去の曜日パターンで12階層平均を改善する", "売上ラグ28〜42日を追加"),
}


def input_hash(*files: str) -> str:
    return files_sha256(DATA / name for name in (*COMMON_FILES, *files))


def forecast(data: Sales, origin: int, run: str) -> np.ndarray:
    group = EXPERIMENTS[run][0]
    return seasonal(data, origin) if group == "seasonal" else predict(data, origin, group)


def experiment_settings(experiment: str) -> tuple[str, dict[str, str | int | float | bool]]:
    group = EXPERIMENTS[experiment][0]
    if group == "seasonal":
        return "seasonal_naive", {"experiment": experiment}
    return "lightgbm", {
        "experiment": experiment,
        **PARAMS,
        "num_boost_round": ROUNDS,
        "first_train_origin": FIRST_TRAIN_ORIGIN,
        "train_stride": STRIDE,
        "series_buckets": SERIES_BUCKETS,
        "feature_group": group,
    }


def run_cv(experiment: str, run_id: str) -> None:
    with STORE.reserve(run_id) as reservation:
        started = time.perf_counter()
        version = code_version(ROOT.parents[1])
        data = load_sales()
        hierarchy = Hierarchy.from_series(data.series)
        if len(hierarchy.labels) != 42_840:
            raise ValueError("公式階層の系列数が一致しません")
        predictions = []
        levels = []
        for origin in ORIGINS:
            scorer = Scorer.at(data, hierarchy, origin)
            prediction = forecast(data, origin, experiment)
            scores = scorer.score(data.actual(origin + 1, origin + HORIZON), prediction)
            predictions.append(prediction)
            levels.append(scores)
            print(f"{experiment} d_{origin}: {scores.mean():.6f}", flush=True)
        windows = np.array(ORIGINS, dtype=np.int32)
        scores = np.vstack(levels)
        group, hypothesis, change = EXPERIMENTS[experiment]
        model, params = experiment_settings(experiment)
        record = ExperimentRecord(
            competition_id="021",
            run_id=run_id,
            metric="WRMSSE",
            greater_is_better=False,
            data_version=group,
            data_hash=input_hash(VALIDATION_FILE),
            split_version=SPLIT_VERSION,
            split_hash=array_sha256(windows),
            seed=SEED,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model=model,
            params=params,
            fold_scores=scores.mean(axis=1).tolist(),
            oof_score=float(scores.mean()),
            runtime_sec=time.perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
        )

        def save(path: Path) -> None:
            np.save(path / "origins.npy", windows)
            np.save(path / "predictions.npy", np.stack(predictions))
            np.save(path / "level_scores.npy", scores)
            data.series.to_csv(path / "series.csv", index=False)

        print(STORE.publish(reservation, record, save), flush=True)


def select(run_ids: list[str]) -> dict:
    expected = set(EXPERIMENTS)
    records = [STORE.record(run_id) for run_id in run_ids]
    if (
        len(records) != len(expected)
        or {r.params.get("experiment") for r in records} != expected
        or any("cv_run" in r.params for r in records)
    ):
        raise ValueError("各実験の CV 記録が1件ずつ必要です")
    data_hash = input_hash(VALIDATION_FILE)
    origins = np.array(ORIGINS, dtype=np.int32)
    split_hash = array_sha256(origins)
    for record in records:
        experiment = record.params["experiment"]
        model, params = experiment_settings(experiment)
        if (
            record.model != model
            or record.params != params
            or record.seed != SEED
            or record.data_version != EXPERIMENTS[experiment][0]
            or (record.competition_id, record.metric, record.greater_is_better)
            != ("021", "WRMSSE", False)
        ):
            raise ValueError(f"{record.run_id}: 実験設定が一致しません")
        if (record.data_hash, record.split_hash, record.split_version) != (
            data_hash,
            split_hash,
            SPLIT_VERSION,
        ):
            raise ValueError(f"{record.run_id}: CV の入力または分割が一致しません")

    data = load_sales()
    hierarchy = Hierarchy.from_series(data.series)
    scorers = [Scorer.at(data, hierarchy, origin) for origin in ORIGINS]
    actual = [data.actual(origin + 1, origin + HORIZON) for origin in ORIGINS]
    comparisons = {}
    for record in records:
        run_id = record.run_id
        if not np.array_equal(STORE.load_array(run_id, "origins.npy"), origins):
            raise ValueError(f"{run_id}: 保存済みの起点が一致しません")
        series = pd.read_csv(STORE.artifact_path(run_id, "series.csv"))
        if not series.equals(data.series):
            raise ValueError(f"{run_id}: 系列 ID または行順が一致しません")
        predictions = STORE.load_array(run_id, "predictions.npy")
        if predictions.shape != (len(ORIGINS), len(data.series), HORIZON):
            raise ValueError(f"{run_id}: 保存済み予測の形状が一致しません")
        scores = np.vstack(
            [
                scorer.score(y, pred)
                for scorer, y, pred in zip(scorers, actual, predictions, strict=True)
            ]
        )
        folds = scores.mean(axis=1).tolist()
        average = float(scores.mean())
        if (
            not np.array_equal(scores, STORE.load_array(run_id, "level_scores.npy"))
            or folds != record.fold_scores
            or average != record.oof_score
        ):
            raise ValueError(f"{run_id}: 再採点と保存済みスコアが一致しません")
        comparisons[run_id] = {
            "wrmsse": average,
            "folds": folds,
            "level_wrmsse": scores.mean(axis=0).tolist(),
        }
    best = min(records, key=lambda r: (comparisons[r.run_id]["wrmsse"], r.params["experiment"]))
    return {
        "selected": best.run_id,
        "experiment": best.params["experiment"],
        "data_hash": data_hash,
        "split_hash": split_hash,
        "comparisons": comparisons,
    }


def finalize(run_id: str, selection_path: Path) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != select(list(selection["comparisons"])):
        raise ValueError("選定結果と CV 記録が一致しません")
    best = STORE.record(selection["selected"])
    with STORE.reserve(run_id) as reservation:
        started = time.perf_counter()
        version = code_version(ROOT.parents[1])
        public = load_sales()
        private = load_sales(evaluation=True)
        if not public.series.equals(private.series) or not np.array_equal(
            public.units, private.units[:, :PUBLIC_ORIGIN]
        ):
            raise ValueError("validation と evaluation の共通履歴が一致しません")
        experiment = selection["experiment"]
        public_pred = forecast(public, PUBLIC_ORIGIN, experiment)
        private_pred = forecast(private, PRIVATE_ORIGIN, experiment)
        csv = submission(public, public_pred, private, private_pred)
        hierarchy = Hierarchy.from_series(public.series)
        scorer = Scorer.at(public, hierarchy, PUBLIC_ORIGIN)
        actual = private.actual(PUBLIC_ORIGIN + 1, PRIVATE_ORIGIN)
        scores = scorer.score(actual, public_pred)
        weekly = np.abs(actual - public_pred).reshape(-1, 4, 7).mean(axis=(0, 2))
        active = (public.actual(PUBLIC_ORIGIN - HORIZON + 1, PUBLIC_ORIGIN) > 0).sum(axis=1)
        absolute = np.abs(actual - public_pred)
        frequency = {}
        for name, mask in (
            ("0", active == 0),
            ("1-7", (active >= 1) & (active <= 7)),
            ("8-21", (active >= 8) & (active <= 21)),
            ("22-28", active >= 22),
        ):
            frequency[name] = {
                "series": int(mask.sum()),
                "mae": float(absolute[mask].mean()) if mask.any() else None,
            }
        diagnostics = {
            "public_wrmsse": float(scores.mean()),
            "public_level_wrmsse": scores.tolist(),
            "public_weekly_mae": weekly.tolist(),
            "public_demand_frequency_mae": frequency,
        }
        record = ExperimentRecord(
            competition_id="021",
            run_id=run_id,
            metric="WRMSSE",
            greater_is_better=False,
            data_version=best.data_version,
            data_hash=input_hash(VALIDATION_FILE, EVALUATION_FILE, "sample_submission.csv"),
            split_version=best.split_version,
            split_hash=best.split_hash,
            seed=best.seed,
            hypothesis="選定済みモデルを公式の2つの締切で再学習する",
            changed_one_thing="CV 選定後の再学習と提出予測",
            model=best.model,
            params={
                **best.params,
                "cv_run": best.run_id,
                "refit_origins": [PUBLIC_ORIGIN, PRIVATE_ORIGIN],
            },
            fold_scores=best.fold_scores,
            oof_score=best.oof_score,
            runtime_sec=time.perf_counter() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
            inputs=STORE.input_hashes(best.run_id, "predictions.npy", "level_scores.npy"),
        )

        def save(output: Path) -> None:
            csv.to_csv(output / "submission.csv", index=False)
            (output / "diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
            )
            np.save(output / "public_predictions.npy", public_pred)
            np.save(output / "private_predictions.npy", private_pred)

        print(STORE.publish(reservation, record, save), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv")
    cv.add_argument("experiments", nargs="+", choices=EXPERIMENTS)
    cv.add_argument("--run-id")
    selection = commands.add_parser("select")
    selection.add_argument("run_ids", nargs="*")
    selection.add_argument("--output", type=Path, default=SELECTION)
    final = commands.add_parser("final")
    final.add_argument("run_id")
    final.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args()
    if args.command == "cv":
        if args.run_id and len(args.experiments) != 1:
            parser.error("--run-id は実験を1件指定する場合に使えます")
        for experiment in args.experiments:
            run_cv(experiment, args.run_id or experiment)
    elif args.command == "select":
        result = select(args.run_ids or list(EXPERIMENTS))
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    else:
        finalize(args.run_id, args.selection)


if __name__ == "__main__":
    main()
