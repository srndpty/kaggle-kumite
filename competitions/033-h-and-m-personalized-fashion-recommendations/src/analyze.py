"""保存 OOF どうしの差を、顧客単位の bootstrap と顧客層別に調べる（選定・export には使わない）。

例: uv run python src/analyze.py E00:E01 E03:E04 --training E04 E08
"""

import argparse

import numpy as np
import polars as pl
from candidates import Snapshot
from data import COMP, FILES, LAST_DAY, Tables, load_tables
from evaluation import (
    HORIZON,
    K,
    Window,
    assessor,
    split_windows,
    validation_windows,
    window_ap,
)
from model import METHODS, label_candidates

from kumite.bootstrap import group_bootstrap_diff
from kumite.provenance import files_sha256
from kumite.quality_gates import verify_cv_run
from kumite.ranking import sparse_hits
from kumite.runs import RunStore

SEGMENTS = ("履歴なし", "直近28日に購入なし", "直近28日に購入あり")


def segment(tables: Tables, window: Window) -> np.ndarray:
    """窓の開始時点での購入者の区分（SEGMENTS の番号）。"""
    last = (
        tables.transactions.filter(pl.col("day") < window.start)
        .group_by("customer")
        .agg(last=pl.col("day").max())
    )
    days = (
        pl.DataFrame({"customer": window.customers}, schema={"customer": pl.Int32})
        .join(last, on="customer", how="left", maintain_order="left")["last"]
        .fill_null(-1)
        .to_numpy()
    )
    return np.where(days < 0, 0, np.where(days >= window.start - 28, 2, 1))


def hit_counts(tables: Tables, window: Window, ranked: np.ndarray) -> tuple[int, int, int]:
    """上位 K 件の（再購入商品の枠数, 再購入の命中, それ以外の命中）。"""
    target = pl.DataFrame({"customer": window.customers}, schema={"customer": pl.Int32})
    past = (
        tables.transactions.filter(pl.col("day") < window.start)
        .join(target.with_row_index("row"), on="customer")
        .select("row", "article")
        .unique()
    )
    known = sparse_hits(
        ranked, past["row"].to_numpy(), past["article"].to_numpy(), tables.n_articles
    )
    hits = sparse_hits(ranked, window.rows, window.items, tables.n_articles)
    return int(known.sum()), int((hits & known).sum()), int((hits & ~known).sum())


def main(pairs: list[tuple[str, str]]) -> None:
    tables = load_tables()
    windows = validation_windows(tables.transactions, LAST_DAY)
    assess = assessor(windows, tables.n_articles, files_sha256(FILES))
    store = RunStore(COMP)
    customers = np.concatenate([w.customers for w in windows])
    segments = np.concatenate([segment(tables, w) for w in windows])
    ap: dict[str, np.ndarray] = {}
    for run_id in dict.fromkeys(run_id for pair in pairs for run_id in pair):
        verify_cv_run(store, run_id, assess)
        parts = split_windows(windows, store.load_array(run_id, "oof.npy"))
        ap[run_id] = np.concatenate(
            [window_ap(w, p, tables.n_articles) for w, p in zip(windows, parts, strict=True)]
        )
        counts = np.sum(
            [hit_counts(tables, w, p) for w, p in zip(windows, parts, strict=True)], axis=0
        )
        by_segment = " ".join(
            f"{name} {ap[run_id][segments == i].mean():.5f}（{np.mean(segments == i):.1%}）"
            for i, name in enumerate(SEGMENTS)
        )
        print(f"{run_id} MAP@12 {ap[run_id].mean():.6f} {by_segment}")
        print(
            f"  上位12件の再購入枠 {counts[0]}（全 {len(customers) * K}） "
            f"再購入の命中 {counts[1]} それ以外の命中 {counts[2]}"
        )
    for base, other in pairs:
        # MAP は大きいほど良いので、損失は 1 - AP。差は (other - base) の MAP に直して表示する
        diff = group_bootstrap_diff(
            1 - ap[base], 1 - ap[other], customers, transform=lambda mean: mean
        )
        print(f"{base} → {other}: {-diff.point:+.6f} 95%区間 [{-diff.high:+.6f}, {-diff.low:+.6f}]")


def print_training(names: list[str]) -> None:
    """手法ごとに、教師に使う顧客と正例の構成を窓別に表示する（学習はしない）。"""
    tables = load_tables()
    for name in names:
        method = METHODS[name]
        for window in validation_windows(tables.transactions, LAST_DAY):
            for week in range(method.label_weeks, 0, -1):
                start = window.start - HORIZON * week
                snapshot = Snapshot.at(tables.transactions, tables.articles, start, method.rule)
                *_, stats = label_candidates(tables, snapshot, method.label_pool)
                shown = " ".join(f"{key} {value:.5g}" for key, value in stats.items())
                print(f"{name} 窓 {window.start} {shown}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="*", help="基準:比較対象")
    parser.add_argument("--training", nargs="+", default=[], choices=METHODS, help="手法")
    args = parser.parse_args()
    if args.pairs:
        main([(base, other) for base, other in (pair.split(":") for pair in args.pairs)])
    print_training(args.training)
