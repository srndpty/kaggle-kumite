"""run の比較とエラー分析。

- クラスタ単位 bootstrap: 類似配列クラスタを再標本化し、MCRMSE の差の 95% 区間を出す
- 外部評価: 各 run の fold 平均 test 予測を公開済み Private 正解で採点する（選定には使わない）
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from data import (
    COMPETITION,
    PRIVATE_LABELS,
    PRIVATE_SCORED,
    SCORED,
    TRAIN_SCORED,
    kmer_clusters,
    mcrmse,
    read_train,
    write_submission,
)

from kumite.runs import RunStore

N_BOOTSTRAP = 2000


def cluster_bootstrap(base: str, other: str, store: RunStore) -> tuple[float, float, float, float]:
    """(other - base) の点推定・95% 区間・差が負の割合。採点対象（SN_filter=1）の分子のみ。"""
    molecules, labels = read_train()
    y = labels.scored()
    keep = labels.sn_filter
    clusters = kmer_clusters(molecules.sequence)[keep]
    a = store.load_array(base, "oof.npy")[keep]
    b = store.load_array(other, "oof.npy")[keep]
    y = y[keep]
    # 分子ごと・列ごとの二乗誤差の和（位置数は全分子で同じ）
    sq_a = ((a - y) ** 2).sum(axis=1)
    sq_b = ((b - y) ** 2).sum(axis=1)
    codes, uniques = pd.factorize(clusters)
    sums_a = np.stack([np.bincount(codes, sq_a[:, k], len(uniques)) for k in range(3)], 1)
    sums_b = np.stack([np.bincount(codes, sq_b[:, k], len(uniques)) for k in range(3)], 1)
    counts = np.bincount(codes, minlength=len(uniques)) * TRAIN_SCORED

    def score(sums: np.ndarray, n: np.ndarray) -> np.ndarray:
        return np.sqrt(sums / n[..., None]).mean(axis=-1)

    total = np.asarray(counts.sum())
    point = float(score(sums_b.sum(0), total) - score(sums_a.sum(0), total))
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)), N_BOOTSTRAP)
    diff = score(draws @ sums_b, draws @ counts) - score(draws @ sums_a, draws @ counts)
    low, high = np.quantile(diff, [0.025, 0.975])
    return point, float(low), float(high), float((diff < 0).mean())


def private_truth() -> tuple[list[str], np.ndarray]:
    truth = pd.read_csv(PRIVATE_LABELS)
    truth = truth[truth["test_filter"] == 1]
    y = np.stack([np.array([json.loads(v)[:PRIVATE_SCORED] for v in truth[c]]) for c in SCORED], -1)
    return list(truth["id"]), y


def private_breakdown(run_id: str, store: RunStore) -> dict[str, float]:
    """Private の全体・学習で見た位置（0〜67）・外挿位置（68〜90）の MCRMSE。"""
    ids, y = private_truth()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "submission.csv"
        write_submission(path, store.load_array(run_id, "test.npy"))
        submission = pd.read_csv(path).set_index("id_seqpos")
    keys = [f"{i}_{p}" for i in ids for p in range(PRIVATE_SCORED)]
    predicted = submission.loc[keys, list(SCORED)].to_numpy().reshape(y.shape)
    return {
        "all": mcrmse(y, predicted),
        "seen_0_67": mcrmse(y[:, :TRAIN_SCORED], predicted[:, :TRAIN_SCORED]),
        "extrapolated_68_90": mcrmse(y[:, TRAIN_SCORED:], predicted[:, TRAIN_SCORED:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    store = RunStore(COMPETITION)
    for run_id in [args.base, *args.runs]:
        record = store.record(run_id)
        parts = private_breakdown(run_id, store)
        text = " ".join(f"{k} {v:.5f}" for k, v in parts.items())
        print(f"{run_id}: CV {record.oof_score:.5f} | Private {text}")
    for run_id in args.runs:
        if store.record(run_id).split_hash != store.record(args.base).split_hash:
            continue
        point, low, high, share = cluster_bootstrap(args.base, run_id, store)
        print(
            f"{run_id} - {args.base}: {point:+.5f} 95%区間 [{low:+.5f}, {high:+.5f}] 負の割合 {share:.3f}"
        )


if __name__ == "__main__":
    main()
