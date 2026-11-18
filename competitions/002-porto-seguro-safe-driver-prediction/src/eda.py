"""上位解法を参照せず、基本分布と分割根拠を記録する。"""

import json

import pandas as pd
from cv import make_folds
from dataset import COMPETITION_DIR, feature_rows, load_data


def main() -> None:
    train, test, _ = load_data()
    x, xt = feature_rows(train), feature_rows(test)
    y = train["target"].to_numpy()
    folds, version = make_folds(x, y)
    out = COMPETITION_DIR / "outputs" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(x.columns),
        "positive_rate": float(y.mean()),
        "duplicate_feature_rows": int(x.duplicated().sum()),
        "train_test_id_overlap": int(train["id"].isin(test["id"]).sum()),
        "split_version": version,
        "fold_positive_rates": [float(y[folds == k].mean()) for k in range(5)],
    }
    stats = pd.DataFrame(
        {
            "dtype": x.dtypes.astype(str),
            "unique": x.nunique(),
            "train_missing_rate": ((x == -1) | x.isna()).mean(),
            "test_missing_rate": ((xt == -1) | xt.isna()).mean(),
            "train_mean": x.mean(),
            "test_mean": xt.mean(),
        }
    )
    stats.to_csv(out / "columns.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
