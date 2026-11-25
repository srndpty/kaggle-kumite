"""主要な分布を表示する（探索用。学習・選定・export のソースには含めない）。"""

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"
PAIR = ["customer_id", "article_id"]


def main() -> None:
    tx = pl.read_csv(
        DATA / "transactions_train.csv",
        schema_overrides={"article_id": pl.String, "t_dat": pl.Date},
    )
    customers = pl.read_csv(DATA / "customers.csv")
    articles = pl.read_csv(DATA / "articles.csv", schema_overrides={"article_id": pl.String})
    sample = pl.read_csv(DATA / "sample_submission.csv")
    last = tx["t_dat"].max()
    print("取引", tx.height, tx["t_dat"].min(), last, tx.null_count().to_dicts())
    print("顧客", customers.height, "商品", articles.height, "提出", sample.height)
    print("提出と顧客の順序が一致", sample["customer_id"].equals(customers["customer_id"]))
    print("顧客の欠損", customers.null_count().to_dicts())
    print("取引のある顧客", tx["customer_id"].n_unique(), "商品", tx["article_id"].n_unique())

    tx = tx.with_columns(week=(pl.lit(last) - pl.col("t_dat")).dt.total_days() // 7)
    for week in range(5):
        past = tx.filter(pl.col("week") > week)
        pairs = tx.filter(pl.col("week") == week).select(PAIR).unique()
        buyers = pairs.select("customer_id").unique()
        recent = past.filter(pl.col("week") <= week + 4)

        def share(history: pl.DataFrame, keys: list[str], rows: pl.DataFrame) -> float:
            return (
                rows.join(history.select(keys).unique(), on=keys, how="semi").height / rows.height
            )

        ranked = (
            past.filter(pl.col("week") == week + 1)
            .group_by("article_id")
            .len()
            .sort(["len", "article_id"], descending=[True, False])
        )
        cover = {
            k: round(share(ranked.head(k), ["article_id"], pairs), 4) for k in (12, 100, 500, 2000)
        }
        print(
            f"末尾から{week}週前: 購入者 {buyers.height} 購入ペア {pairs.height}"
            f" 1人あたり {pairs.height / buyers.height:.2f}"
            f" 履歴なし顧客 {1 - share(past, ['customer_id'], buyers):.4f}"
            f" 直近4週に購入した顧客 {share(recent, ['customer_id'], buyers):.4f}"
            f" 再購入ペア {share(past, PAIR, pairs):.4f} 直近4週 {share(recent, PAIR, pairs):.4f}"
            f" 初売り商品のペア {1 - share(past, ['article_id'], pairs):.4f} 前週上位の被覆 {cover}"
        )


if __name__ == "__main__":
    main()
