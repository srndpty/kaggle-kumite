"""公式データの規模、極端値、取引の観測範囲を集計する。"""

import polars as pl
from dataset import ID, TARGET, TRANSACTION_FILES, load_cards
from train import EXTREME, ROOT


def main() -> None:
    data = ROOT / "data"
    train, test, _ = load_cards(data)
    y = train[TARGET]
    print(f"train={len(train)} test={len(test)} extreme={int((y < EXTREME).sum())}")
    print(y.describe(percentiles=[0.01, 0.99]).to_string())
    for label, cards in (("train", train), ("test", test)):
        print(label, cards["first_active_month"].str[:4].value_counts(dropna=False).to_dict())
    for name in TRANSACTION_FILES:
        rows = pl.scan_csv(data / name)
        print(
            name,
            rows.select(
                pl.len(),
                pl.col(ID).n_unique().alias("cards"),
                pl.col("month_lag").min().alias("min_month"),
                pl.col("month_lag").max().alias("max_month"),
            )
            .collect()
            .to_dicts(),
            rows.group_by(ID)
            .len()
            .select(
                pl.col("len").median().alias("median_transactions"),
                pl.col("len").max().alias("max_transactions"),
            )
            .collect()
            .to_dicts(),
        )


if __name__ == "__main__":
    main()
