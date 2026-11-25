"""締切前の取引だけから候補を作り、規則の順に並べる。

並びは「本人の直近購入 → 共購入 → 人気」。同じ商品は最初に現れた位置だけを残す。
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl

from kumite.ranking import validate_full_rankings

RECENT_DAYS = 28
POPULAR_DAYS = 7
MAX_OWN = 30
MAX_NEIGHBOURS = 50
MAX_BASKET = 50
MIN_COUNT = 2
MAX_COVISIT = 50
WIDTH = 100
_NO_NEIGHBOURS = pl.DataFrame(
    schema={"article": pl.Int32, "neighbour": pl.Int32, "weight": pl.Float64}
)


@dataclass(frozen=True)
class Rule:
    """候補の作り方。"""

    own: bool
    half_life: float | None  # None なら直近 POPULAR_DAYS 日の件数
    covisit: bool


@dataclass(frozen=True)
class Snapshot:
    """ある締切の時点で見える取引。作るときに締切以降を切り落とす。"""

    cutoff: int
    history: pl.DataFrame
    rule: Rule

    @classmethod
    def at(cls, transactions: pl.DataFrame, cutoff: int, rule: Rule) -> "Snapshot":
        return cls(cutoff, transactions.filter(pl.col("day") < cutoff), rule)

    def last_days(self, days: int) -> pl.DataFrame:
        return self.history.filter(pl.col("day") >= self.cutoff - days)

    @cached_property
    def recent(self) -> pl.DataFrame:
        return self.last_days(RECENT_DAYS)

    @cached_property
    def popular(self) -> pl.DataFrame:
        """人気上位 WIDTH 件（article, popular_rank）。"""
        if self.rule.half_life is None:
            scored = self.last_days(POPULAR_DAYS).group_by("article").agg(score=pl.len())
        else:
            age = self.cutoff - 1 - pl.col("day")
            scored = (
                self.recent.with_columns(w=0.5 ** (age / self.rule.half_life))
                .group_by("article")
                .agg(score=pl.col("w").sort().sum())
            )
        top = scored.sort(["score", "article"], descending=[True, False]).head(WIDTH)
        if top.height != WIDTH:
            raise ValueError("人気候補が不足しています")
        return top.select("article").with_row_index("popular_rank")

    @cached_property
    def neighbours(self) -> pl.DataFrame:
        """直近の共購入の条件付き割合（article → neighbour, weight）。"""
        pairs = self.recent.select("customer", "article").unique()
        counts = pairs.group_by("article").agg(n_article=pl.len())
        size = pl.len().over("customer")
        baskets = pairs.filter((size >= 2) & (size <= MAX_BASKET))
        return (
            baskets.join(baskets.rename({"article": "neighbour"}), on="customer")
            .filter(pl.col("article") != pl.col("neighbour"))
            .group_by("article", "neighbour")
            .agg(n=pl.len())
            .filter(pl.col("n") >= MIN_COUNT)
            .join(counts, on="article")
            .with_columns(weight=pl.col("n") / pl.col("n_article"))
            .sort(["article", "weight", "neighbour"], descending=[False, True, False])
            .group_by("article", maintain_order=True)
            .head(MAX_NEIGHBOURS)
            .select("article", "neighbour", "weight")
        )

    def own(self, customers: pl.DataFrame) -> pl.DataFrame:
        """本人の直近購入（新しい順、own_rank・own_days・own_count）。"""
        return (
            self.recent.join(customers, on="customer", how="semi")
            .group_by("customer", "article")
            .agg(last=pl.col("day").max(), own_count=pl.len())
            .sort(
                ["customer", "last", "own_count", "article"],
                descending=[False, True, True, False],
            )
            .with_columns(
                own_rank=pl.int_range(pl.len()).over("customer"),
                own_days=self.cutoff - pl.col("last"),
            )
            .filter(pl.col("own_rank") < MAX_OWN)
            .drop("last")
        )

    def covisit(self, own: pl.DataFrame) -> pl.DataFrame:
        """本人の直近購入と一緒に買われた商品（covisit_rank・covisit_score）。"""
        neighbours = self.neighbours if self.rule.covisit else _NO_NEIGHBOURS
        return (
            own.select("customer", "article")
            .join(neighbours, on="article")
            .group_by("customer", pl.col("neighbour").alias("article"))
            # 浮動小数の和は足す順で変わるので、並べてから足す
            .agg(covisit_score=pl.col("weight").sort().sum())
            .sort(["customer", "covisit_score", "article"], descending=[False, True, False])
            .with_columns(covisit_rank=pl.int_range(pl.len()).over("customer"))
            .filter(pl.col("covisit_rank") < MAX_COVISIT)
        )

    def candidates(self, customers: np.ndarray) -> pl.DataFrame:
        """顧客ごとに WIDTH 件（customer・position 順）。候補源の値が無ければ null。"""
        target = pl.DataFrame({"customer": customers}, schema={"customer": pl.Int32})
        own = self.own(target)
        if not self.rule.own:
            own = own.clear()
        covisit = self.covisit(own)
        sources = [
            own.select("customer", "article", source=pl.lit(0), order=pl.col("own_rank")),
            covisit.select("customer", "article", source=pl.lit(1), order=pl.col("covisit_rank")),
            target.join(self.popular, how="cross").select(
                "customer",
                "article",
                source=pl.lit(2),
                order=pl.col("popular_rank").cast(pl.Int64),
            ),
        ]
        merged = (
            pl.concat(sources)
            .sort("customer", "source", "order")
            .unique(["customer", "article"], keep="first", maintain_order=True)
            .with_columns(position=pl.int_range(pl.len()).over("customer"))
            .filter(pl.col("position") < WIDTH)
        )
        return (
            merged.join(own, on=["customer", "article"], how="left")
            .join(covisit, on=["customer", "article"], how="left")
            .join(self.popular, on="article", how="left")
            .sort("customer", "position")
        )


def candidate_matrix(
    candidates: pl.DataFrame, customers: np.ndarray, n_articles: int
) -> np.ndarray:
    """（顧客, 位置）の商品番号。customers は昇順。"""
    if candidates.height != len(customers) * WIDTH:
        raise ValueError("候補の件数が顧客数と一致しません")
    ids = candidates["customer"].to_numpy().reshape(-1, WIDTH)
    if not (ids == customers[:, None]).all():
        raise ValueError("候補の顧客の順序が一致しません")
    matrix = candidates["article"].to_numpy().reshape(-1, WIDTH)
    validate_full_rankings(matrix, (len(customers), WIDTH), n_articles)
    return matrix
