"""BNP Paribas の欠損処理を固定 CV で比較する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from dummy import ROOT
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

SEED = 42
FILES = ("train.csv", "test.csv", "sample_submission.csv")
VARIANTS = {
    "E00": ("基準", "数値中央値・カテゴリ欠損"),
    "E01": ("木の欠損分岐が有効", "数値のネイティブ欠損処理"),
    "E02": ("欠損の有無が予測に効く", "数値欠損指示"),
    "E04": ("時点の差に情報がある", "v40 と v50 の単位を概算して差を追加"),
    "E03": ("カテゴリ欠損と未知カテゴリは異なる", "カテゴリ欠損を独立カテゴリ化"),
}


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=4,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def load(data: Path):
    train, test, sample = (pd.read_csv(data / name) for name in FILES)
    if list(sample.columns) != ["ID", "PredictedProb"]:
        raise ValueError("提出列が不正です")
    for frame in (train, test, sample):
        if frame.empty or frame["ID"].hasnans or not frame["ID"].is_unique:
            raise ValueError("ID が空、欠損、または重複しています")
    if set(train["target"]) != {0, 1} or set(train["ID"]) & set(test["ID"]):
        raise ValueError("目的変数または学習・テスト ID が不正です")
    x = train.drop(columns=["ID", "target"])
    query = test.set_index("ID").loc[sample["ID"]]
    if set(query.columns) != set(x.columns) or len(test) != len(sample):
        raise ValueError("テストの特徴列または ID が一致しません")
    return x, train["target"].to_numpy(), query[x.columns].reset_index(drop=True), sample


def transform(train: pd.DataFrame, apply: pd.DataFrame, variant: str):
    train = train.copy()
    apply = apply.copy()
    if variant == "E04":
        for frame in (train, apply):
            frame["observation_minus_elapsed"] = frame["v40"] - (0.00072 / 0.00147) * frame["v50"]
    numeric = train.select_dtypes(include="number").columns
    categorical = train.columns.difference(numeric, sort=False)
    for name in categorical:
        levels = pd.Index(train[name].dropna().unique())
        dtype = pd.CategoricalDtype(levels)
        train[name] = train[name].astype(dtype)
        original = apply[name]
        apply[name] = original.where(original.isin(levels)).astype(dtype)
        if variant == "E03" and train[name].isna().any():
            marker = "__missing__"
            if marker in levels or original.eq(marker).any():
                raise ValueError("カテゴリ欠損用の予約値が入力にあります")
            train[name] = train[name].cat.add_categories([marker]).fillna(marker)
            apply[name] = apply[name].cat.add_categories([marker]).mask(original.isna(), marker)
    if variant == "E02":
        missing = numeric[train[numeric].isna().any()]
        train = pd.concat([train, train[missing].isna().add_suffix("_missing")], axis=1)
        apply = pd.concat([apply, apply[missing].isna().add_suffix("_missing")], axis=1)
    if variant != "E01":
        medians = train[numeric].median()
        train[numeric] = train[numeric].fillna(medians)
        apply[numeric] = apply[numeric].fillna(medians)
    return train, apply


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y, pred, labels=[0, 1]))


def run(data: Path, run_id: str, variant: str) -> None:
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        x, y, query, sample = load(data)
        groups = pd.util.hash_pandas_object(x, index=False)
        if groups.is_unique:
            splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))
        else:
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
            splits = list(splitter.split(x, y, groups))
        roles = split_roles(len(y), splits)

        def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
            rows = pd.concat([x.iloc[valid], query], ignore_index=True)
            train_x, apply_x = transform(x.iloc[train], rows, variant)
            model = make_model().fit(train_x, y[train])
            probabilities = np.asarray(model.predict_proba(apply_x))
            validate_probabilities(probabilities, len(rows), 2)
            return FoldOutput(probabilities[: len(valid), 1], probabilities[len(valid) :, 1])

        def artifacts(out: Path) -> None:
            np.save(out / "split_roles.npy", roles)
            sample.assign(PredictedProb=result.query).to_csv(out / "submission.csv", index=False)

        start = perf_counter()
        result = cross_fit_splits(len(y), splits, fit)
        validate_probabilities(np.column_stack([1 - result.query, result.query]), len(sample), 2)
        publish_cv_run(
            store,
            reservation,
            Evaluation(
                "010",
                "LogLoss",
                False,
                score,
                files_sha256(data / name for name in FILES),
                "stratified-duplicates-v1",
                array_sha256(roles),
                SEED,
            ),
            Trial(
                *VARIANTS[variant],
                "LightGBM",
                "kaggle-v1",
                {"variant": variant, **make_model().get_params()},
            ),
            y,
            splits,
            result,
            perf_counter() - start,
            artifacts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eda", action="store_true")
    mode.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.eda:
        x, y, query, _ = load(args.data)
        print(
            json.dumps(
                {
                    "train_shape": x.shape,
                    "test_shape": query.shape,
                    "positive_rate": float(y.mean()),
                    "numeric": len(x.select_dtypes(include="number").columns),
                    "train_missing": int(x.isna().sum().sum()),
                    "test_missing": int(query.isna().sum().sum()),
                    "train_rows_with_missing": int(x.isna().any(axis=1).sum()),
                    "duplicate_rows": int(x.duplicated().sum()),
                },
                indent=2,
            )
        )
    else:
        run(args.data, args.run_id or args.variant, args.variant)


if __name__ == "__main__":
    main()
