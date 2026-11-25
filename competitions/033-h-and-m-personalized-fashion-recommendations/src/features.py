"""再ランキングの特徴量。締切前の取引と、商品属性・年齢だけを使う。"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl
from candidates import Rule, Snapshot
from data import CATEGORIES, Tables

HISTORY_DAYS = 365
AFFINITIES = ["product_code", "product_type_no", "department_no", "garment_group_no"]
STATIC = [name for name in CATEGORIES if name != "product_code"]
FEATURES = [
    "position",
    "own_rank",
    "own_days",
    "own_count",
    "covisit_rank",
    "covisit_score",
    "popular_rank",
    "item_1",
    "item_3",
    "item_7",
    "item_28",
    "item_trend",
    "item_age",
    "item_price",
    "item_online",
    "item_buyer_age",
    "age",
    "age_gap",
    "customer_7",
    "customer_28",
    "customer_365",
    "customer_recency",
    "customer_price",
    "customer_online",
    "price_ratio",
    *(f"affinity_{name}" for name in AFFINITIES),
    *(f"share_{name}" for name in AFFINITIES),
    *STATIC,
]


def feature_names(rule: Rule) -> list[str]:
    """候補源を増やした規則だけ、その順位を特徴に加える。"""
    return [*FEATURES, *(["product_rank"] if rule.product else [])]


@dataclass(frozen=True)
class FeatureBuilder:
    tables: Tables
    snapshot: Snapshot

    @cached_property
    def items(self) -> pl.DataFrame:
        cutoff = self.snapshot.cutoff
        recent = self.snapshot.recent.join(self.tables.ages, on="customer", how="left")
        counts = [
            pl.col("day").filter(pl.col("day") >= cutoff - d).len().alias(f"item_{d}")
            for d in (1, 3, 7, 28)
        ]
        stats = recent.group_by("article").agg(
            *counts,
            item_price=pl.col("price").sort().mean(),
            item_online=(pl.col("channel") == 2).mean(),
            item_buyer_age=pl.col("age").sort().mean(),
        )
        first = self.snapshot.history.group_by("article").agg(item_age=cutoff - pl.col("day").min())
        return stats.join(first, on="article").with_columns(
            item_trend=pl.col("item_7") / pl.col("item_28")
        )

    @cached_property
    def year(self) -> pl.DataFrame:
        return self.snapshot.last_days(HISTORY_DAYS).join(
            self.tables.articles.select("article", *AFFINITIES), on="article"
        )

    def customers(self, history: pl.DataFrame) -> pl.DataFrame:
        cutoff = self.snapshot.cutoff
        return history.group_by("customer").agg(
            customer_7=pl.col("day").filter(pl.col("day") >= cutoff - 7).len(),
            customer_28=pl.col("day").filter(pl.col("day") >= cutoff - 28).len(),
            customer_365=pl.len(),
            customer_recency=cutoff - pl.col("day").max(),
            customer_price=pl.col("price").sort().mean(),
            customer_online=(pl.col("channel") == 2).mean(),
        )

    def matrix(self, candidates: pl.DataFrame) -> np.ndarray:
        """candidates と同じ行順の feature_names 行列。無い値は NaN。"""
        target = candidates.select("customer").unique()
        history = self.year.join(target, on="customer", how="semi")
        frame = (
            candidates.lazy()
            .join(self.items.lazy(), on="article", how="left", maintain_order="left")
            .join(self.customers(history).lazy(), on="customer", how="left", maintain_order="left")
            .join(self.tables.ages.lazy(), on="customer", how="left", maintain_order="left")
            .join(self.tables.articles.lazy(), on="article", how="left", maintain_order="left")
        )
        for name in AFFINITIES:
            counts = history.group_by("customer", name).agg(pl.len().alias(f"affinity_{name}"))
            frame = frame.join(
                counts.lazy(), on=["customer", name], how="left", maintain_order="left"
            ).with_columns(
                (pl.col(f"affinity_{name}") / pl.col("customer_365")).alias(f"share_{name}")
            )
        result = (
            frame.with_columns(
                age_gap=(pl.col("age") - pl.col("item_buyer_age")).abs(),
                price_ratio=pl.col("item_price") / pl.col("customer_price"),
            )
            .select(pl.col(feature_names(self.snapshot.rule)).cast(pl.Float32))
            .collect()
        )
        if result.height != candidates.height:
            raise ValueError("特徴量の結合で候補の行数が変わりました")
        return result.to_numpy()
