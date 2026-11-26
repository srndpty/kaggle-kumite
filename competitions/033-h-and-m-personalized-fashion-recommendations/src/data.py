"""公式 CSV を整数 ID の表として読む。"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
FILES = [
    DATA / name
    for name in ("articles.csv", "customers.csv", "sample_submission.csv", "transactions_train.csv")
]
LAST_DAY = 733  # 2020-09-22（2018-09-20 からの日数）
CATEGORIES = [
    "product_code",
    "product_type_no",
    "department_no",
    "garment_group_no",
    "index_group_no",
    "colour_group_code",
]


@dataclass(frozen=True)
class Tables:
    transactions: pl.DataFrame  # day, customer, article, price, channel
    articles: pl.DataFrame  # article（通し番号）と CATEGORIES
    ages: pl.DataFrame  # customer, age
    customer_ids: pl.Series  # 提出順
    article_ids: pl.Series

    @property
    def n_articles(self) -> int:
        return len(self.article_ids)


def load_tables() -> Tables:
    raw_articles = pl.read_csv(
        DATA / "articles.csv",
        columns=["article_id", *CATEGORIES],
        schema_overrides={"article_id": pl.String},
    )
    customers = pl.read_csv(DATA / "customers.csv", columns=["customer_id", "age"])
    sample = pl.read_csv(DATA / "sample_submission.csv", columns=["customer_id"])
    if not sample["customer_id"].equals(customers["customer_id"]):
        raise ValueError("提出対象の顧客と顧客表の順序が一致しません")
    for frame, key in ((raw_articles, "article_id"), (customers, "customer_id")):
        if not frame[key].is_unique().all():
            raise ValueError(f"{key} が一意ではありません")
    articles = raw_articles.with_row_index("article").with_columns(pl.col("article").cast(pl.Int32))
    indexed = customers.with_row_index("customer").with_columns(pl.col("customer").cast(pl.Int32))
    raw = pl.read_csv(
        DATA / "transactions_train.csv",
        schema_overrides={"article_id": pl.String, "t_dat": pl.Date, "price": pl.Float64},
    )
    transactions = (
        raw.join(indexed.select("customer_id", "customer"), on="customer_id", how="inner")
        .join(articles.select("article_id", "article"), on="article_id", how="inner")
        .select(
            (pl.col("t_dat") - pl.date(2018, 9, 20)).dt.total_days().cast(pl.Int32).alias("day"),
            "customer",
            "article",
            "price",
            pl.col("sales_channel_id").cast(pl.Int32).alias("channel"),
        )
        .sort("day", "customer", "article", "price", "channel")
    )
    if transactions.height != raw.height or transactions["day"].max() != LAST_DAY:
        raise ValueError("取引の件数または最終日が想定と一致しません")
    return Tables(
        transactions,
        articles.drop("article_id").with_columns(pl.col(CATEGORIES).cast(pl.Int32)),
        indexed.select("customer", pl.col("age").cast(pl.Float32)),
        customers["customer_id"],
        raw_articles["article_id"],
    )
