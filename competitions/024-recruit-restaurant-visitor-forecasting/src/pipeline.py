"""実験の定義・実行・保存。train.py・leak_check.py などで共通に使う。"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from cv import SEED, SPLITS
from dataset import (
    COMPETITION_DIR,
    DATA_FILES,
    DATE,
    STORE,
    TARGET,
    load_air_stores,
    load_date_info,
    load_reserves,
    load_test,
    load_visits,
)
from features import Feature, FoldContext, holiday_adjusted_dow, row_features
from models import MODELS

from kumite.checks import validate_regression
from kumite.crossfit import (
    CrossFitResult,
    FoldOutput,
    Split,
    cross_fit_splits,
    mean_rounds,
    split_roles,
)
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import Reservation, RunStore

COMPETITION_ID = "024"
METRIC = "rmsle"
STORE_RUNS = RunStore(COMPETITION_DIR)

# 目的変数の変換: 名前 → (変換, 逆変換)
TARGETS: dict[
    str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
] = {
    "raw": (lambda y: y, lambda p: p),
    "log1p": (np.log1p, np.expm1),
}


@dataclass(frozen=True)
class Experiment:
    model: str
    hypothesis: str
    changed_one_thing: str
    features: tuple[Feature, ...] = ()  # 分割ごとに作る特徴
    target: str = "raw"  # TARGETS の名前
    split: str = "time"  # cv.SPLITS の名前
    refit: bool = False  # テスト予測を、全学習行で学習し直した 1 モデルで作る
    # 曜日: 暦どおりか、祝日を土曜・休日の前日を金曜・翌日を月曜とみなすか（特徴の dow と、統計量の曜日）
    dow: Literal["calendar", "holiday_adjusted"] = "calendar"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def data_version(self) -> str:
        names = ["base", *(f.name for f in self.features)]
        submission = "refit" if self.refit else "split_mean"
        dow = "" if self.dow == "calendar" else f"|dow={self.dow}"
        return f"{'+'.join(names)}|target={self.target}|submission={submission}{dow}"


@dataclass(frozen=True)
class Data:
    train: pd.DataFrame  # 来店記録（日付・店舗の昇順）
    test: pd.DataFrame  # 提出と同じ行順
    x_train: pd.DataFrame  # 行の属性だけで決まる特徴量
    x_test: pd.DataFrame
    reserves: pd.DataFrame
    holidays: pd.Series  # 日付 → 祝日フラグ

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=np.float64)

    def splits(self, name: str) -> list[Split]:
        return SPLITS[name][1](self.train, self.test[STORE].unique())


def load_data() -> Data:
    train, test, stores = load_visits(), load_test(), load_air_stores()
    dates = load_date_info()
    return Data(
        train,
        test,
        row_features(train, stores),
        row_features(test, stores),
        load_reserves(),
        dates.set_index("calendar_date")["holiday_flg"],
    )


def rmsle(y: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error（log1p 空間の RMSE）。"""
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y)) ** 2)))


def fold_inputs(
    exp: Experiment, data: Data, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1 分割の (学習, 検証, テスト) の入力を作る。ラベルを使う特徴には学習行の来店数だけを渡す。"""
    train, valid = data.train.iloc[train_idx], data.train.iloc[valid_idx]
    test_origin, test_end = data.test[DATE].min(), data.test[DATE].max()
    # 予定が分かるのは予測期間の末尾まで。学習・検証行は検証窓の末尾まで（全行で学習し直すときは
    # 検証行が無いので test の末尾まで）
    fit_end = valid[DATE].max() if len(valid) else test_end
    rows = [train, valid, data.test]
    if exp.dow == "holiday_adjusted":
        ends = [fit_end, fit_end, test_end]
        dows = [
            holiday_adjusted_dow(r[DATE], data.holidays, e) for r, e in zip(rows, ends, strict=True)
        ]
    else:
        dows = [r[DATE].dt.day_of_week.to_numpy() for r in rows]
    train, valid, test = (r.assign(dow=d) for r, d in zip(rows, dows, strict=True))
    frames = [
        x.assign(dow=d)
        for x, d in zip(
            (data.x_train.iloc[train_idx], data.x_train.iloc[valid_idx], data.x_test),
            dows,
            strict=True,
        )
    ]
    ctx = FoldContext(
        train=train,
        valid=valid,
        test=test,
        log_y=np.log1p(y[train_idx]),
        reserves=data.reserves,
        holidays=data.holidays,
        valid_origin=valid[DATE].min() if len(valid) else test_origin,
        test_origin=test_origin,
        fit_known_end=fit_end,
        test_known_end=test_end,
    )
    for feature in exp.features:
        built = feature.build(ctx)
        frames = [
            pd.concat([x, pd.DataFrame(v, columns=feature.columns, index=x.index)], axis=1)
            for x, v in zip(frames, built, strict=True)
        ]
    return frames[0], frames[1], frames[2]


def fit_split(
    exp: Experiment, data: Data, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> FoldOutput:
    """1 分割を学習し、来店数の尺度に戻した予測（0 以上）を返す。"""
    forward, inverse = TARGETS[exp.target]
    x_tr, x_va, x_te = fold_inputs(exp, data, y, train_idx, valid_idx)
    out = MODELS[exp.model](
        x_tr, forward(y[train_idx]), x_va, forward(y[valid_idx]), x_te, exp.params
    )
    return FoldOutput(
        np.maximum(inverse(out.valid), 0), np.maximum(inverse(out.query), 0), out.info
    )


def run_cv(exp: Experiment, data: Data, splits: list[Split]) -> CrossFitResult:
    """splits で CV し、OOF 予測とテスト予測を返す。

    テスト予測は分割ごとのモデルの平均。exp.refit なら、全学習行で学習し直した 1 モデルの予測
    （round 数は分割ごとの best iteration の平均）。CV の値は変わらない。
    """
    y = data.y
    result = cross_fit_splits(len(y), splits, lambda tr, va: fit_split(exp, data, y, tr, va))
    if not exp.refit:
        return result
    rounds = mean_rounds(result.infos)
    fixed = replace(exp, params={**exp.params, "num_boost_round": rounds, "early_stopping": None})
    no_rows = np.array([], dtype=np.int64)
    query = fit_split(fixed, data, y, np.arange(len(y)), no_rows).query
    infos = [{**info, "refit_rounds": rounds} for info in result.infos]
    return CrossFitResult(result.oof, query, infos)


def save_run(
    reservation: Reservation,
    exp: Experiment,
    data: Data,
    splits: list[Split],
    result: CrossFitResult,
    runtime_sec: float,
) -> None:
    """予測を検査してからスコアを計算し、成果物と実験ログを保存する（kumite.cv_run）。"""
    y = data.y
    validated = np.concatenate([valid for _, valid in splits])
    validate_regression(result.oof[validated], len(validated), lower=0)
    validate_regression(result.query, len(data.test), lower=0)
    evaluation = Evaluation(
        competition_id=COMPETITION_ID,
        metric=METRIC,
        greater_is_better=False,
        score=rmsle,
        data_hash=files_sha256(DATA_FILES),
        split_version=SPLITS[exp.split][0],
        split_hash=array_sha256(split_roles(len(y), splits)),
        seed=SEED,
    )
    trial = Trial(
        hypothesis=exp.hypothesis,
        changed_one_thing=exp.changed_one_thing,
        model=exp.model,
        data_version=exp.data_version,
        params=exp.params,
    )

    def write_submission(out_dir: Path) -> None:
        frame = pd.DataFrame({"id": data.test["id"].to_numpy(), TARGET: result.query})
        frame.to_csv(out_dir / "submission.csv", index=False)

    publish_cv_run(
        STORE_RUNS, reservation, evaluation, trial, y, splits, result, runtime_sec, write_submission
    )


def load_predictions(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(OOF 予測, テスト予測) を、ログのハッシュと照合して返す。"""
    return STORE_RUNS.load_array(run_id, "oof.npy"), STORE_RUNS.load_array(run_id, "test.npy")
