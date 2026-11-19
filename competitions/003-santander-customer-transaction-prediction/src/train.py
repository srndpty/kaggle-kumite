"""Santander のローカル診断、ダミー作成、CV と提出ファイル作成。"""

import argparse
import json
import re
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from features import transform
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
ID = "ID_code"
TARGET = "target"
VARIANTS = {
    "E01": ("ベースライン", "生値の標準化", "L2 ロジスティック回帰"),
    "E02": ("反復する値に予測情報がある", "出現回数を追加", "L2 ロジスティック回帰"),
    "E03": ("列ごとの非線形な主効果がある", "分位点ビンに置換", "L2 ロジスティック回帰"),
    "E04": ("木でも非線形な主効果を捉えられる", "深さ 1 の LightGBM", "LightGBM"),
    "E05": ("交互作用の寄与は小さい", "木の深さを 3 に変更", "LightGBM"),
    "E06": (
        "反復する値と未出現値で値の効果が異なる",
        "反復値だけを残した特徴を追加",
        "L2 ロジスティック回帰",
    ),
}


def read_submission(data: Path) -> pd.DataFrame:
    sample = pd.read_csv(data / "sample_submission.csv", dtype={ID: str})
    if list(sample.columns) != [ID, TARGET] or sample.empty:
        raise ValueError("提出雛形の列または行数が不正です")
    if sample[ID].hasnans or not sample[ID].is_unique:
        raise ValueError("提出雛形の ID に欠損または重複があります")
    return sample


def data_hash(data: Path) -> str:
    return files_sha256(data / name for name in ("train.csv", "test.csv", "sample_submission.csv"))


def load_data(data: Path):
    train = pd.read_csv(data / "train.csv", dtype={ID: str})
    test = pd.read_csv(data / "test.csv", dtype={ID: str})
    sample = read_submission(data)
    columns = train.columns.drop([ID, TARGET])
    if not len(columns) or set(test.columns) != {ID, *columns}:
        raise ValueError("学習・テストの特徴列が一致しません")
    if train[ID].hasnans or not train[ID].is_unique or set(train[ID]) & set(test[ID]):
        raise ValueError("学習 ID に欠損または重複があります")
    if not test[ID].is_unique or set(test[ID]) != set(sample[ID]):
        raise ValueError("テスト ID が提出雛形と一致しません")
    if set(train[TARGET]) != {0, 1}:
        raise ValueError("target に二値以外の値があるか、片方のクラスがありません")
    x = train.loc[:, columns].to_numpy(dtype=np.float64)
    query = test.set_index(ID).loc[sample[ID], columns].to_numpy(dtype=np.float64)
    return x, train[TARGET].to_numpy(dtype=np.int8), query, sample, train[ID].to_numpy()


def row_groups(x: np.ndarray) -> np.ndarray:
    # float64 の入力精度で完全一致を判定する。
    return np.unique(x, axis=0, return_inverse=True)[1]


def make_splits(x: np.ndarray, y: np.ndarray, seed: int):
    groups = row_groups(x)
    if len(np.unique(groups)) == len(y):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y))
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y, groups))
    if any(len(np.unique(y[rows])) != 2 for pair in splits for rows in pair):
        raise ValueError("ROC AUC の計算には各 fold の両クラスが必要です")
    return splits


def diagnose(x: np.ndarray, y: np.ndarray, query: np.ndarray, ids: np.ndarray) -> None:
    order = np.argsort([int(str(value).rsplit("_", 1)[-1]) for value in ids])
    unique_counts = [np.unique(col, return_counts=True)[1] for col in x.T]
    info = {
        "train_shape": x.shape,
        "test_shape": query.shape,
        "positive_rate": float(y.mean()),
        "train_missing": int(np.isnan(x).sum()),
        "train_infinite": int(np.isinf(x).sum()),
        "test_missing": int(np.isnan(query).sum()),
        "test_infinite": int(np.isinf(query).sum()),
        "constant_columns": sum(len(counts) == 1 for counts in unique_counts),
        "duplicate_rows": len(x) - len(np.unique(row_groups(x))),
        "singleton_fraction_per_column": [
            float(np.count_nonzero(counts == 1) / len(x)) for counts in unique_counts
        ],
        "positive_rate_by_id_decile": [
            float(y[rows].mean()) for rows in np.array_split(order, 10) if len(rows)
        ],
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred))


def model_params(variant: str, seed: int) -> dict[str, int | float | str | bool]:
    if variant in ("E04", "E05"):
        return dict(
            objective="binary",
            n_estimators=600,
            learning_rate=0.1,
            num_leaves=8,
            max_depth=1 if variant == "E04" else 3,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    return dict(C=1.0, max_iter=500, solver="lbfgs", random_state=seed)


def fold_predict(
    variant: str,
    params: dict,
    train: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
) -> FoldOutput:
    apply_rows = np.concatenate((valid, query))
    if variant in ("E04", "E05"):
        model = LGBMClassifier(**params)
        # LightGBM が fit 時と推論時に同じ特徴名を参照する。
        model.fit(pd.DataFrame(train), y)
        apply_x = pd.DataFrame(apply_rows)
    else:
        model = LogisticRegression(**params)
        fit_x, apply_x = transform(variant, train, apply_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(fit_x, y)
    start = perf_counter()
    proba = np.asarray(model.predict_proba(apply_x))
    predict_sec = perf_counter() - start
    validate_probabilities(proba, len(valid) + len(query), 2)
    return FoldOutput(proba[: len(valid), 1], proba[len(valid) :, 1], {"predict_sec": predict_sec})


def write_submission(sample: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    validate_probabilities(np.column_stack((1 - pred, pred)), len(sample), 2)
    sample.assign(target=pred).to_csv(path, index=False)


def run(data: Path, run_id: str, variant: str, seed: int) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"未知の variant です: {variant}")
    match = re.match(r"^(E0[1-6])(?:$|[-_])", run_id)
    if match and match[1] != variant:
        raise ValueError(f"run ID {run_id} と variant {variant} が一致しません")
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        x, y, query, sample, _ = load_data(data)
        if not np.isfinite(x).all() or not np.isfinite(query).all():
            raise ValueError("NaN または Inf があります。EDA で原因を確認してください")
        splits = make_splits(x, y, seed)
        roles = split_roles(len(y), splits)
        if not np.all((roles == 2).sum(axis=0) == 1):
            raise ValueError("OOF が全学習行を 1 回ずつカバーしていません")
        evaluation = Evaluation(
            "003",
            "roc_auc",
            True,
            score,
            data_hash(data),
            f"stratified5-v1-seed{seed}",
            array_sha256(roles),
            seed,
        )
        params = model_params(variant, seed)
        trial = Trial(
            *VARIANTS[variant], data_version="kaggle-v1", params={"variant": variant, **params}
        )

        def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
            return fold_predict(variant, params, x[train], y[train], x[valid], query)

        start = perf_counter()
        result = cross_fit_splits(len(y), splits, fit_fold)
        validate_probabilities(np.column_stack((1 - result.oof, result.oof)), len(y), 2)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            perf_counter() - start,
            lambda out: write_submission(sample, result.query, out / "submission.csv"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id")
    mode.add_argument("--eda", action="store_true")
    mode.add_argument("--dummy", type=Path, metavar="OUTPUT")
    parser.add_argument("--variant", choices=VARIANTS, default="E01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dummy is not None:
        sample = read_submission(args.data)
        write_submission(sample, np.full(len(sample), 0.5), args.dummy)
    elif args.eda:
        x, y, query, _, ids = load_data(args.data)
        diagnose(x, y, query, ids)
    else:
        run(args.data, args.run_id, args.variant, args.seed)


if __name__ == "__main__":
    main()
