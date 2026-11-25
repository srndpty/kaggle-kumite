"""7日間の検証窓・正解・OOF 採点。評価単位は（窓, 窓内に購入した顧客）。"""

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.quality_gates import Artifact, AssessOOF, Scores
from kumite.ranking import average_precision, sparse_hits, validate_full_rankings

K = 12
HORIZON = 7
N_WINDOWS = 3


@dataclass(frozen=True)
class Window:
    start: int
    customers: np.ndarray  # 昇順
    rows: np.ndarray  # customers の位置
    items: np.ndarray

    @property
    def n_relevant(self) -> np.ndarray:
        return np.bincount(self.rows, minlength=len(self.customers))


def window_truth(transactions: pl.DataFrame, start: int) -> Window:
    pairs = (
        transactions.filter(pl.col("day").is_between(start, start + HORIZON - 1))
        .select("customer", "article")
        .unique()
        .sort("customer", "article")
    )
    customers, rows = np.unique(pairs["customer"].to_numpy(), return_inverse=True)
    return Window(start, customers, rows, pairs["article"].to_numpy())


def validation_windows(transactions: pl.DataFrame, last_day: int) -> list[Window]:
    """末尾から連続する 7 日窓（古い順）。"""
    return [window_truth(transactions, last_day + 1 - HORIZON * k) for k in range(N_WINDOWS, 0, -1)]


def split_hash(windows: list[Window]) -> str:
    units = [np.column_stack([np.full(len(w.customers), w.start), w.customers]) for w in windows]
    return array_sha256(np.concatenate(units).astype(np.int64))


def window_ap(window: Window, ranked: np.ndarray, n_articles: int) -> np.ndarray:
    hits = sparse_hits(ranked, window.rows, window.items, n_articles)
    return average_precision(hits, window.n_relevant)


def split_windows(windows: list[Window], oof: np.ndarray) -> list[np.ndarray]:
    return np.split(oof, np.cumsum([len(w.customers) for w in windows])[:-1])


def scores(windows: list[Window], oof: np.ndarray, n_articles: int) -> Scores:
    """窓別の MAP@12 と、全（窓, 顧客）をまとめた MAP@12。"""
    ap = [
        window_ap(window, part, n_articles)
        for window, part in zip(windows, split_windows(windows, oof), strict=True)
    ]
    return [float(x.mean()) for x in ap], float(np.concatenate(ap).mean())


def assessor(windows: list[Window], n_articles: int, data_hash: str) -> AssessOOF:
    expected_split = split_hash(windows)
    total = sum(len(w.customers) for w in windows)

    def assess(artifact: Artifact, record: ExperimentRecord) -> Scores:
        if record.data_hash != data_hash or record.split_hash != expected_split:
            raise ValueError("データまたは検証分割のハッシュが記録と一致しません")
        oof = np.load(artifact("oof.npy"), allow_pickle=False)
        validate_full_rankings(oof, (total, K), n_articles)
        json.loads(artifact("diagnostics.json").read_text(encoding="utf-8"))
        return scores(windows, oof, n_articles)

    return assess


def candidate_metrics(window: Window, candidates: np.ndarray, n_articles: int) -> dict[str, float]:
    """候補 Recall（ペア単位）と、候補内の正解を先頭に並べた場合の MAP@12。"""
    found = sparse_hits(candidates, window.rows, window.items, n_articles).sum(axis=1)
    bound = np.minimum(found, K) / np.minimum(window.n_relevant, K)
    return {
        "recall": float(found.sum() / len(window.items)),
        "map_upper_bound": float(bound.mean()),
        "customers_with_hit": float((found > 0).mean()),
    }
