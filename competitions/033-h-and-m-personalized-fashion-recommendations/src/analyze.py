"""保存 OOF どうしの差を、顧客単位の bootstrap と顧客層別に調べる（選定・export には使わない）。

例: uv run python src/analyze.py E00:E01 E03:E04
"""

import argparse

import numpy as np
import polars as pl
from data import COMP, FILES, LAST_DAY, Tables, load_tables
from evaluation import (
    Window,
    assessor,
    split_windows,
    validation_windows,
    window_ap,
)

from kumite.bootstrap import group_bootstrap_diff
from kumite.provenance import files_sha256
from kumite.quality_gates import verify_cv_run
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
        by_segment = " ".join(
            f"{name} {ap[run_id][segments == i].mean():.5f}（{np.mean(segments == i):.1%}）"
            for i, name in enumerate(SEGMENTS)
        )
        print(f"{run_id} MAP@12 {ap[run_id].mean():.6f} {by_segment}")
    for base, other in pairs:
        # MAP は大きいほど良いので、損失は 1 - AP。差は (other - base) の MAP に直して表示する
        diff = group_bootstrap_diff(
            1 - ap[base], 1 - ap[other], customers, transform=lambda mean: mean
        )
        print(f"{base} → {other}: {-diff.point:+.6f} 95%区間 [{-diff.high:+.6f}, {-diff.low:+.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="+", help="基準:比較対象")
    args = parser.parse_args()
    main([(base, other) for base, other in (pair.split(":") for pair in args.pairs)])
