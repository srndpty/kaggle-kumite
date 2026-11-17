"""実験の定義・実行・保存。train.py・leak_check.py などで共通に使う。"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cv import SEED, SPLITS
from dataset import (
    COMPETITION_DIR,
    DATA_FILES,
    TARGET,
    load_store,
    load_test,
    load_train,
    modeling_rows,
)
from features import GAP_COLUMNS, Calendar, Gaps, SupervisedFeature, row_features
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

COMPETITION_ID = "019"
METRIC = "rmspe"
STORE = RunStore(COMPETITION_DIR)

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
    gaps: Gaps | None = None  # 休業日・祝日までの日数（分割ごとに作る。ラベルは使わない）
    supervised: SupervisedFeature | None = None  # ラベルを使う特徴（分割ごとに作る）
    target: str = "raw"  # TARGETS の名前
    split: str = "time"  # cv.SPLITS の名前
    refit: bool = False  # テスト予測を、全学習行で学習し直した 1 モデルで作る
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def data_version(self) -> str:
        names = ["base", *(f.name for f in (self.gaps, self.supervised) if f is not None)]
        submission = "refit" if self.refit else "split_mean"
        return f"{'+'.join(names)}|target={self.target}|submission={submission}"


@dataclass(frozen=True)
class Data:
    train: pd.DataFrame  # 売上 > 0 の行（日付・店舗の昇順）
    test: pd.DataFrame  # 提出と同じ行順
    x_train: pd.DataFrame  # 行の属性だけで決まる特徴量
    x_test: pd.DataFrame
    calendar: Calendar

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=np.float64)

    def splits(self, name: str) -> list[Split]:
        return SPLITS[name][1](self.train, self.test["Store"].unique())


def load_data() -> Data:
    train_all, test, store = load_train(), load_test(), load_store()
    train = modeling_rows(train_all)
    return Data(
        train,
        test,
        row_features(train, store),
        row_features(test, store),
        Calendar.build(train_all, test),
    )


def pseudo_test(data: Data, rows: np.ndarray) -> Data:
    """train の rows（末尾の期間）を test とみなした Data。提出時と同じ情報の条件にそろえる。

    - 暦は rows の末尾までに切り詰める（提出時は test の末尾より先の予定が分からないのと同じ）
    - rows より後の train 行があれば拒否する（その売上も提出時には存在しない）
    """
    dates = data.train["Date"]
    as_of = dates.iloc[rows].max()
    if (dates > as_of).any():
        raise ValueError(f"疑似テストの末尾 {as_of.date()} より後の学習行があります")
    held = replace(
        data,
        test=data.train.iloc[rows],
        x_test=data.x_train.iloc[rows],
        calendar=data.calendar.until(as_of),
    )
    assert held.calendar.end == as_of
    return held


def rmspe(y: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Square Percentage Error。売上 0 の行は評価しない（学習・検証行には含めていない）。"""
    return float(np.sqrt(np.mean(((y - pred) / y) ** 2)))


def _with_columns(x: pd.DataFrame, values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.concat([x, pd.DataFrame(values, columns=columns, index=x.index)], axis=1)


def fold_inputs(
    exp: Experiment, data: Data, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1 分割の (学習, 検証, テスト) の入力を作る。ラベルを使う特徴には学習行の売上だけを渡す。"""
    x_tr, x_va, x_te = data.x_train.iloc[train_idx], data.x_train.iloc[valid_idx], data.x_test
    train, valid = data.train.iloc[train_idx], data.train.iloc[valid_idx]
    if exp.gaps is not None:
        # 予定が分かるのは予測期間の末尾まで。censor なら、学習・検証行は検証窓の末尾まで、
        # テスト行はテストの末尾までの予定だけを見る（全行で学習し直すときは学習行もテストの末尾まで）
        test_end = data.test["Date"].max()
        valid_end = valid["Date"].max() if len(valid) else test_end
        known_fit, known_test = (
            (valid_end, test_end) if exp.gaps.censor else (data.calendar.end, data.calendar.end)
        )
        fit_rows = pd.concat([train, valid])
        g_fit = data.calendar.gaps_for(fit_rows, exp.gaps, known_fit)
        g_te = data.calendar.gaps_for(data.test, exp.gaps, known_test)
        x_tr = _with_columns(x_tr, g_fit[: len(train)], GAP_COLUMNS)
        x_va = _with_columns(x_va, g_fit[len(train) :], GAP_COLUMNS)
        x_te = _with_columns(x_te, g_te, GAP_COLUMNS)
    if exp.supervised is None:
        return x_tr, x_va, x_te
    rest = pd.concat([valid, data.test], ignore_index=True)
    f_tr, f_rest = exp.supervised(train, np.log1p(y[train_idx]), rest)
    columns = exp.supervised.columns
    return (
        _with_columns(x_tr, f_tr, columns),
        _with_columns(x_va, f_rest[: len(valid_idx)], columns),
        _with_columns(x_te, f_rest[len(valid_idx) :], columns),
    )


def fit_split(
    exp: Experiment, data: Data, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> FoldOutput:
    """1 分割を学習し、売上の尺度に戻した予測（0 以上）を返す。"""
    forward, inverse = TARGETS[exp.target]
    x_tr, x_va, x_te = fold_inputs(exp, data, y, train_idx, valid_idx)
    out = MODELS[exp.model](
        x_tr, forward(y[train_idx]), x_va, forward(y[valid_idx]), x_te, exp.params
    )
    return FoldOutput(
        np.maximum(inverse(out.valid), 0), np.maximum(inverse(out.query), 0), out.info
    )


def refit(
    exp: Experiment,
    data: Data,
    y: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    rounds: int,
) -> FoldOutput:
    """train_idx の行で、検証行を見ずに rounds まで学習する。valid は predict_idx の行への予測。"""
    fixed = replace(exp, params={**exp.params, "num_boost_round": rounds, "early_stopping": None})
    return fit_split(fixed, data, y, train_idx, predict_idx)


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
    no_rows = np.array([], dtype=np.int64)
    query = refit(exp, data, y, np.arange(len(y)), no_rows, rounds).query
    infos = [{**info, "refit_rounds": rounds} for info in result.infos]
    return CrossFitResult(result.oof, query, infos)


def submission_frame(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """休業日（Open = 0）は 0 にする。Open が欠損の行は営業として予測をそのまま出す。"""
    sales = np.where(test["Open"].to_numpy() == 0, 0.0, pred)
    return pd.DataFrame({"Id": test["Id"].to_numpy(), TARGET: sales})


def save_run(
    reservation: Reservation,
    exp: Experiment,
    data: Data,
    splits: list[Split],
    result: CrossFitResult,
    runtime_sec: float,
    inputs: dict[str, str] | None = None,
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
        score=rmspe,
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
        inputs=inputs or {},
    )

    def write_submission(out_dir: Path) -> None:
        submission_frame(data.test, result.query).to_csv(out_dir / "submission.csv", index=False)

    publish_cv_run(
        STORE, reservation, evaluation, trial, y, splits, result, runtime_sec, write_submission
    )


def load_predictions(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(OOF 予測, テスト予測) を、ログのハッシュと照合して返す。"""
    return STORE.load_array(run_id, "oof.npy"), STORE.load_array(run_id, "test.npy")
