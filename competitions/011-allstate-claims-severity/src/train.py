"""Allstate の EDA と目的変数・学習損失の比較。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from dummy import ROOT, submission
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from kumite.checks import validate_regression
from kumite.crossfit import FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import lightgbm_regression
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

FILES = ("train.csv", "test.csv", "sample_submission.csv")
SEED = 42
CATEGORIES = [f"cat{i}" for i in range(1, 117)]
CONTINUOUS = [f"cont{i}" for i in range(1, 15)]
N_CENTERS = 16
VARIANTS = {
    "E00": ("元尺度 L2 が定数中央値を上回る", "元尺度 L2"),
    "E01": ("元尺度 L1 で全体 MAE が下がる", "目的関数を L1 に変更"),
    "E02": ("中央付近の過大予測が減り、上位 1% の過小予測が増える", "log1p 変換"),
    "E03": ("上位 1% の過小予測が E02 より小さく、中央の過大予測が E00 より少ない", "平方根変換"),
    "E04": ("連続値の同時分布を表現すると MAE が下がる", "クラスタ中心までの距離を追加"),
}


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (pd.read_csv(data / name, keep_default_na=False) for name in FILES)
    y = train["loss"].to_numpy(dtype=np.float64)
    validate_regression(y, len(train), lower=0)
    ids = train["id"]
    if ids.empty or ids.hasnans or not ids.is_unique or set(ids) & set(test["id"]):
        raise ValueError("学習 ID が不正です")
    expected = {"id", *CATEGORIES, *CONTINUOUS}
    if set(train.columns) != expected | {"loss"} or set(test.columns) != expected:
        raise ValueError("特徴列が一致しません")
    submission(sample, test["id"], np.zeros(len(test)))
    return train, test, sample


def groups(x: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(x, index=False).to_numpy()


def make_splits(x: pd.DataFrame) -> tuple[str, list[Split]]:
    keys = groups(x)
    if len(np.unique(keys)) == len(keys):
        return "shuffled-5fold-v1", list(KFold(5, shuffle=True, random_state=SEED).split(x))
    splitter = GroupKFold(5, shuffle=True, random_state=SEED)
    return "duplicate-group-5fold-v1", list(splitter.split(x, groups=keys))


def features(
    training: pd.DataFrame, apply: pd.DataFrame, categories: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, apply = training.copy(), apply.copy()
    for name in categories:
        dtype = pd.CategoricalDtype(training[name].dropna().unique())
        training[name] = training[name].astype(dtype)
        apply[name] = apply[name].where(apply[name].isin(dtype.categories)).astype(dtype)
    return training, apply


def target(y: np.ndarray, variant: str) -> np.ndarray:
    if variant == "E02":
        return np.log1p(y)
    if variant == "E03":
        return np.sqrt(y)
    return y


def add_distances(training: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler().fit(training[CONTINUOUS])
    train_scaled = scaler.transform(training[CONTINUOUS])
    apply_scaled = scaler.transform(apply[CONTINUOUS])
    columns = [f"center_distance_{i}" for i in range(N_CENTERS)]
    model = MiniBatchKMeans(n_clusters=N_CENTERS, random_state=SEED)
    # 型スタブは n_init を文字列に限定しているが、実装は整数も受け付ける。
    model.set_params(n_init=3)
    with threadpool_limits(limits=4):
        model.fit(train_scaled)
        train_distances = pd.DataFrame(
            model.transform(train_scaled), columns=columns, index=training.index
        )
        apply_distances = pd.DataFrame(
            model.transform(apply_scaled), columns=columns, index=apply.index
        )
    return pd.concat([training, train_distances], axis=1), pd.concat(
        [apply, apply_distances], axis=1
    )


def restore(pred: np.ndarray, variant: str, count: int) -> np.ndarray:
    if variant == "E02":
        pred = np.expm1(pred)
    elif variant == "E03":
        pred = np.square(pred)
    validate_regression(pred, count)
    return np.maximum(pred, 0)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def eda(data: Path) -> None:
    train, test, _ = load(data)
    x = train.drop(columns=["id", "loss"])
    y = train["loss"]
    blocks = []
    for block in np.array_split(train.sort_values("id").index.to_numpy(), 5):
        rows = train.loc[block]
        ids = rows["id"]
        test_count = test["id"].between(ids.min(), ids.max()).sum()
        blocks.append(
            {
                "median_loss": float(rows["loss"].median()),
                "mean_loss": float(rows["loss"].mean()),
                "test_count": int(test_count),
            }
        )
    print(
        json.dumps(
            {
                "train_shape": train.shape,
                "test_shape": test.shape,
                "loss_quantiles": y.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).to_dict(),
                "top_1pct_loss_share": float(
                    y.nlargest(int(np.ceil(len(y) / 100))).sum() / y.sum()
                ),
                "train_missing": int(train.isna().sum().sum()),
                "test_missing": int(test.isna().sum().sum()),
                "duplicate_train_rows": int(x.duplicated().sum()),
                "duplicate_train_test_rows": int(np.isin(groups(test[x.columns]), groups(x)).sum()),
                "test_only_category_levels": sum(
                    len(set(test[c]) - set(train[c])) for c in CATEGORIES
                ),
                "id_blocks": blocks,
                "split": make_splits(x)[0],
            },
            indent=2,
        )
    )


def run(data: Path, variant: str, run_id: str) -> None:
    store = RunStore(ROOT)
    params = {
        "objective": "regression_l1" if variant == "E01" else "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "num_boost_round": 500,
        "early_stopping": None,
        "num_threads": 4,
        "seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
    }
    with store.reserve(run_id) as reservation:
        train, test, sample = load(data)
        y = train["loss"].to_numpy(dtype=np.float64)
        x = train.drop(columns=["id", "loss"])
        query = test[x.columns]
        version, splits = make_splits(x)
        roles = split_roles(len(y), splits)

        def fit(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
            apply = pd.concat([x.iloc[va], query], ignore_index=True)
            train_x, apply_x = features(x.iloc[tr], apply, CATEGORIES)
            if variant == "E04":
                train_x, apply_x = add_distances(train_x, apply_x)
            prediction = lightgbm_regression(
                train_x,
                target(y[tr], variant),
                apply_x.iloc[: len(va)],
                target(y[va], variant),
                apply_x.iloc[len(va) :],
                params,
                CATEGORIES,
            )
            return FoldOutput(
                restore(prediction.valid, variant, len(va)),
                restore(prediction.query, variant, len(query)),
                {**prediction.info, "median_baseline_mae": mae(y[va], np.median(y[tr]))},
            )

        started = perf_counter()
        result = cross_fit_splits(len(y), splits, fit)
        validate_regression(result.oof, len(y), lower=0)
        frame = submission(sample, test["id"], result.query)

        def artifacts(out: Path) -> None:
            np.save(out / "split_roles.npy", roles)
            frame.to_csv(out / "submission.csv", index=False)

        publish_cv_run(
            store,
            reservation,
            Evaluation(
                "011",
                "MAE",
                False,
                mae,
                files_sha256(data / f for f in FILES),
                version,
                array_sha256(roles),
                SEED,
            ),
            Trial(
                *VARIANTS[variant],
                "LightGBM",
                "kaggle-root-csv-v1",
                {
                    "variant": variant,
                    **params,
                    **({"n_centers": N_CENTERS} if variant == "E04" else {}),
                },
            ),
            y,
            splits,
            result,
            perf_counter() - started,
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
        eda(args.data)
    else:
        run(args.data, args.variant, args.run_id or args.variant)


if __name__ == "__main__":
    main()
