"""ラベル・欠損・キー・相対日時を診断する。上位解法は参照しない。"""

import json

import pandas as pd
from dataset import COMPETITION_DIR, DATA_DIR, KEY, load_data
from features import INSTALLMENT_VALUES, application_features, installment_value_summary


def main() -> None:
    train, test, _ = load_data()
    output = COMPETITION_DIR / "outputs" / "eda"
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate": float(train["TARGET"].mean()),
        "duplicate_feature_rows": int(application_features(train).duplicated().sum()),
        "train_test_common_feature_rows": int(
            pd.util.hash_pandas_object(application_features(train), index=False)
            .isin(pd.util.hash_pandas_object(application_features(test), index=False))
            .sum()
        ),
        "history": {},
    }
    pd.DataFrame(
        {"train": train.drop(columns="TARGET").isna().mean(), "test": test.isna().mean()}
    ).to_csv(output / "missing.csv")
    train.select_dtypes(include="number").describe().to_csv(output / "numeric_train.csv")
    test.select_dtypes(include="number").describe().to_csv(output / "numeric_test.csv")
    histories = {}
    for name, key, times in (
        ("bureau", "SK_ID_BUREAU", ["DAYS_CREDIT", "DAYS_CREDIT_UPDATE"]),
        ("previous_application", "SK_ID_PREV", ["DAYS_DECISION"]),
        ("installments_payments", "SK_ID_PREV", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]),
    ):
        frame = pd.read_csv(DATA_DIR / f"{name}.csv", usecols=[KEY, key, *times])
        counts = frame.groupby(KEY).size()
        histories[name] = {
            "rows": len(frame),
            "customers": int(frame[KEY].nunique()),
            "key_missing": int(frame[[KEY, key]].isna().any(axis=1).sum()),
            "key_duplicate_rows": int(frame[key].duplicated().sum()),
            "contract_multi_customer": int(frame.groupby(key)[KEY].nunique().gt(1).sum()),
            "train_coverage": float(train[KEY].isin(counts.index).mean()),
            "test_coverage": float(test[KEY].isin(counts.index).mean()),
            "history_count_quantiles": {
                str(q): float(counts.quantile(q)) for q in (0, 0.5, 0.9, 0.99, 1)
            },
            "times": {
                column: {
                    "missing": int(frame[column].isna().sum()),
                    "positive": int(frame[column].gt(0).sum()),
                    "min": float(frame[column].min()),
                    "max": float(frame[column].max()),
                }
                for column in times
            },
        }
        counts.rename("history_count").to_csv(output / f"{name}_counts.csv")
    summary["history"] = histories
    summary["installment_values"] = installment_value_summary(
        pd.read_csv(DATA_DIR / "installments_payments.csv", usecols=INSTALLMENT_VALUES)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
