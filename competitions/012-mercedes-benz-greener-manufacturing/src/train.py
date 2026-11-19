"""車両構成をグループとする CV で試験時間を予測する。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from kumite.checks import validate_regression
from kumite.crossfit import FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
SPLIT_VERSION = "group-v1"
INTERACTION = ("X314", "X315")
VARIANTS = {
    "interaction": ("二値特徴の交互作用で改善する", "X314 と X315 の積を追加"),
    "baseline": ("加法モデルの基準", "基準"),
    "rare": ("希少水準をまとめると改善する", "5件未満の水準を統合"),
    "strong": ("正則化を強めると改善する", "alpha=100"),
    "categorical": ("二値特徴を除くと悪化する", "二値特徴を除く"),
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if (
        list(sample.columns) != ["ID", "y"]
        or not all(
            frame["ID"].notna().all() and frame["ID"].is_unique for frame in (train, test, sample)
        )
        or set(test["ID"]) != set(sample["ID"])
        or set(train["ID"]) & set(test["ID"])
    ):
        raise ValueError("提出サンプルと train/test の ID または提出列が不正です")
    test = test.set_index("ID").loc[sample["ID"]]
    x = train.drop(columns=["ID", "y"])
    y = train["y"].to_numpy(dtype=np.float64)
    if list(x.columns) != list(test.columns) or x.isna().any().any() or test.isna().any().any():
        raise ValueError("train/test の特徴列が不一致、または欠損があります")
    validate_binary_features(x, test)
    validate_regression(y, len(x))
    return x, test, y, sample


def make_splits(x: pd.DataFrame) -> tuple[np.ndarray, list[Split]]:
    groups = x.groupby(list(x.columns), sort=False, dropna=False).ngroup().to_numpy()
    splits = list(GroupKFold(n_splits=5).split(x, groups=groups))
    return groups, splits


def categorical_columns(x: pd.DataFrame) -> list[str]:
    return list(x.select_dtypes(exclude="number").columns)


def validate_binary_features(x: pd.DataFrame, test: pd.DataFrame) -> None:
    categories = categorical_columns(x)
    if categories != categorical_columns(test):
        raise ValueError("train/test のカテゴリ列が一致しません")
    binary = [name for name in x.columns if name not in categories]
    for name, frame in (("train", x), ("test", test)):
        invalid = frame[binary].columns[~frame[binary].isin([0, 1]).all()]
        if len(invalid):
            raise ValueError(f"{name} の数値特徴に 0/1 以外の値があります: {list(invalid)}")


def fit_fold(
    x: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    variant: str,
) -> FoldOutput:
    cat = categorical_columns(x)
    binary = [col for col in x.columns if col not in cat]
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5 if variant == "rare" else None)
    preprocessor = ColumnTransformer(
        [("category", encoder, cat)],
        remainder="drop" if variant == "categorical" else "passthrough",
        sparse_threshold=1.0,
    )
    alpha = 100 if variant == "strong" else 10
    model = make_pipeline(preprocessor, Ridge(alpha=alpha, solver="lsqr"))
    model.fit(x.iloc[train_idx], y[train_idx])
    pred = np.asarray(model.predict(x.iloc[valid_idx]))
    return FoldOutput(
        valid=pred,
        query=np.asarray(model.predict(test)),
        info={"mae": float(mean_absolute_error(y[valid_idx], pred)), "n_binary": len(binary)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    variant: str = args.variant
    x, test, y, sample = load_data()
    _, splits = make_splits(x)
    if variant == "interaction":
        a, b = INTERACTION
        x = pd.concat([x, (x[a] * x[b]).rename("interaction")], axis=1)
        test = pd.concat([test, (test[a] * test[b]).rename("interaction")], axis=1)
    evaluation = Evaluation(
        competition_id=ROOT.name,
        metric="r2",
        greater_is_better=True,
        score=r2_score,
        data_hash=files_sha256(
            DATA / name for name in ("train.csv", "test.csv", "sample_submission.csv")
        ),
        split_version=SPLIT_VERSION,
        split_hash=array_sha256(split_roles(len(x), splits)),
        seed=SEED,
    )
    hypothesis, change = VARIANTS[variant]
    trial = Trial(
        hypothesis=hypothesis,
        changed_one_thing=change,
        model="Ridge",
        data_version="kaggle-v1",
        params={"variant": variant, "alpha": 100 if variant == "strong" else 10, "solver": "lsqr"},
    )

    def fit(train_idx: np.ndarray, valid_idx: np.ndarray) -> FoldOutput:
        return fit_fold(x, test, y, train_idx, valid_idx, variant)

    store = RunStore(ROOT)
    with store.reserve(args.run_id) as reservation:
        start = perf_counter()
        result = cross_fit_splits(len(x), splits, fit)
        validate_regression(result.oof, len(x))
        validate_regression(result.query, len(test))
        print(f"OOF MAE {mean_absolute_error(y, result.oof):.5f}")

        def write_submission(out_dir: Path) -> None:
            sample.assign(y=result.query).to_csv(out_dir / "submission.csv", index=False)

        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            perf_counter() - start,
            write_submission,
        )


if __name__ == "__main__":
    main()
