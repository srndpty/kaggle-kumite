"""実験の定義と CV の実行・保存。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cache import CACHE_DIR, SIZE, load_array
from cv import SEED, SPLITS, check_driver_disjoint
from dataset import (
    COMPETITION_DIR,
    DRIVERS_CSV,
    N_CLASSES,
    SUBMISSION_CSV,
    load_test,
    load_train,
    submission,
)
from models import (
    ModelConfig,
    backbone_features,
    finetune,
    linear_probe,
    logloss,
)
from neighbors import features, knn, smooth

from kumite.checks import validate_probabilities
from kumite.crossfit import (
    CrossFitResult,
    FoldOutput,
    Split,
    cross_fit_splits,
    folds_to_splits,
    split_roles,
)
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import Reservation, RunStore

COMPETITION_ID = "060"
METRIC = "multi_logloss"
STORE_RUNS = RunStore(COMPETITION_DIR)
INPUT_SIZE = [SIZE[1], SIZE[0]]  # (高さ, 幅)。縮小キャッシュの大きさのまま入力する


@dataclass(frozen=True)
class Experiment:
    # "prior" / "linear_probe" / "finetune" / "knn_smooth"（他 run の予測の後処理）/
    # "refit"（OOF は base のまま、test は全行で学習し直した 1 モデル）/ "nested_finetune"（nested.py）
    model: str
    hypothesis: str
    changed_one_thing: str
    split: str = "driver"
    config: ModelConfig = field(default_factory=ModelConfig)
    params: dict[str, Any] = field(default_factory=dict[str, Any])  # linear_probe の C など
    base: str | None = None  # 後処理の元になる run_id（knn_smooth）

    @property
    def data_version(self) -> str:
        return f"{CACHE_DIR.name}-q95"

    def logged_params(self) -> dict[str, Any]:
        if self.model == "prior":
            return {}
        if self.model == "knn_smooth":
            return {"base": self.base, **self.params}
        extra = {"base": self.base} if self.base is not None else {}
        cfg = self.config
        if self.model == "linear_probe":
            return {"backbone": cfg.backbone, "size": INPUT_SIZE, **self.params}
        return {
            "backbone": cfg.backbone,
            "size": INPUT_SIZE,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "aug": cfg.aug,
            "hflip": cfg.hflip,
            "seed": cfg.seed,
            **cfg.extra,
            **extra,
            **self.params,
        }


@dataclass(frozen=True)
class Data:
    train: pd.DataFrame
    test: pd.DataFrame
    train_images: np.ndarray  # (枚数, H, W, 3) の uint8（memmap）。行順は train と同じ
    test_images: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return self.train["label"].to_numpy()

    def splits(self, name: str) -> list[Split]:
        folds = SPLITS[name][1](self.train)
        if name == "driver":
            check_driver_disjoint(self.train, folds)
        return folds_to_splits(folds)


def load_data() -> Data:
    train, test = load_train(), load_test()
    images = load_array("train"), load_array("test")
    if (len(images[0]), len(images[1])) != (len(train), len(test)):
        raise ValueError("画像配列の枚数が CSV と一致しません。cache.py を実行し直してください")
    return Data(train, test, *images)


def fit_split(
    exp: Experiment, data: Data, train_idx: np.ndarray, valid_idx: np.ndarray
) -> FoldOutput:
    y = data.y
    print(f"  学習 {len(train_idx)} 枚、検証 {len(valid_idx)} 枚", flush=True)
    if exp.model == "prior":
        prior = np.bincount(y[train_idx], minlength=N_CLASSES) / len(train_idx)
        return FoldOutput(np.tile(prior, (len(valid_idx), 1)), np.tile(prior, (len(data.test), 1)))
    if exp.model == "linear_probe":
        x = backbone_features(exp.config.backbone, data.train_images, "train")
        x_te = backbone_features(exp.config.backbone, data.test_images, "test")
        valid, test, info = linear_probe(
            x[train_idx], y[train_idx], x[valid_idx], x_te, exp.params["C"]
        )
        return FoldOutput(valid, test, info)
    if exp.model == "knn_smooth":
        return _knn_smooth(exp, data, valid_idx)
    if exp.model == "finetune":
        valid, test, info = finetune(
            exp.config,
            data.train_images,
            train_idx,
            y[train_idx],
            valid_idx,
            data.test_images,
            monitor=lambda proba: logloss(y[valid_idx], proba),
        )
        return FoldOutput(valid, test, info)
    raise ValueError(f"未知のモデル: {exp.model}")


def _knn_smooth(exp: Experiment, data: Data, valid_idx: np.ndarray) -> FoldOutput:
    """元の run の予測を、近傍の予測との平均に置き換える。近傍は同じ fold の検証画像の中（test は test の中）。"""
    if exp.base is None:
        raise ValueError("knn_smooth には base（元の run_id）が必要です")
    if STORE_RUNS.record(exp.base).split_version != SPLITS[exp.split][0]:
        raise ValueError(f"{exp.base} は分割が違います（OOF が別の分割の予測になる）")
    oof, test = load_predictions(exp.base)
    kind, k, backbone = exp.params["feature"], exp.params["k"], exp.config.backbone
    x_va = features(kind, "train", data.train_images, backbone)[valid_idx]
    x_te = features(kind, "test", data.test_images, backbone)
    return FoldOutput(smooth(oof[valid_idx], knn(x_va, k)), smooth(test, knn(x_te, k)))


def run_cv(exp: Experiment, data: Data, splits: list[Split]) -> CrossFitResult:
    """splits で CV する。test の予測は分割ごとのモデルの平均（refit は全行で学習し直した 1 モデル）。"""
    if exp.model == "refit":
        return _refit(exp, data, splits)
    return cross_fit_splits(len(data.y), splits, lambda tr, va: fit_split(exp, data, tr, va))


def _refit(exp: Experiment, data: Data, splits: list[Split]) -> CrossFitResult:
    """OOF は base の run のまま（同じ設定の fold モデル）。test は全行・同じ epoch 数で学習し直した 1 モデル。"""
    if exp.base is None:
        raise ValueError("refit には base（OOF を引き継ぐ run_id）が必要です")
    if STORE_RUNS.record(exp.base).split_version != SPLITS[exp.split][0]:
        raise ValueError(f"{exp.base} は分割が違います")
    oof, _ = load_predictions(exp.base)
    all_rows = np.arange(len(data.y))
    print(f"  全行で学習: {len(all_rows)} 枚", flush=True)
    _, test, info = finetune(
        exp.config, data.train_images, all_rows, data.y, all_rows[:0], data.test_images
    )
    return CrossFitResult(oof, test, [info] + [{} for _ in splits[1:]])


def save_run(
    reservation: Reservation,
    exp: Experiment,
    data: Data,
    splits: list[Split],
    result: CrossFitResult,
    runtime_sec: float,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """予測を検査してからスコアを計算し、成果物と実験ログを保存する（kumite.cv_run）。

    extra は追加で保存する配列（ファイル名 → 配列。nested.py の内側の予測など）。
    """
    y = data.y
    validated = np.concatenate([valid for _, valid in splits])
    validate_probabilities(result.oof[validated], len(validated), N_CLASSES)
    validate_probabilities(result.query, len(data.test), N_CLASSES)
    evaluation = Evaluation(
        competition_id=COMPETITION_ID,
        metric=METRIC,
        greater_is_better=False,
        score=logloss,
        data_hash=files_sha256(_data_files()),
        split_version=SPLITS[exp.split][0],
        split_hash=array_sha256(split_roles(len(y), splits)),
        seed=SEED,
    )
    trial = Trial(
        hypothesis=exp.hypothesis,
        changed_one_thing=exp.changed_one_thing,
        model=exp.model,
        data_version=exp.data_version,
        params=exp.logged_params(),
        inputs={} if exp.base is None else STORE_RUNS.input_hashes(exp.base, "oof.npy", "test.npy"),
    )

    def write_submission(out_dir: Path) -> None:
        submission(data.test, result.query).to_csv(out_dir / "submission.csv", index=False)
        for name, array in (extra or {}).items():
            np.save(out_dir / name, array)

    publish_cv_run(
        STORE_RUNS, reservation, evaluation, trial, y, splits, result, runtime_sec, write_submission
    )


def _data_files() -> list[Path]:
    """データの同一性: CSV と、学習・推論に使う縮小キャッシュの全画像。"""
    return [DRIVERS_CSV, SUBMISSION_CSV, *sorted(CACHE_DIR.rglob("*.jpg"))]


def load_predictions(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(OOF の確率, test の確率) を、ログのハッシュと照合して返す。"""
    return STORE_RUNS.load_array(run_id, "oof.npy"), STORE_RUNS.load_array(run_id, "test.npy")
