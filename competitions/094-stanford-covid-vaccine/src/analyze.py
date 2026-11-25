"""run の比較とエラー分析。

- `compare`: 類似配列クラスタを再標本化し、CV の MCRMSE の差の 95% 区間を出す（開発中に使う）
- `private`: 公開済み Private 正解による外部評価。指定した選定計画の記録（outputs/selections）が
  確定している場合だけ実行する。選定前に Private を見て方針を決めないための運用上の鍵
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from data import COMPETITION, TRAIN_SCORED, kmer_clusters, read_train, write_submission
from external import private_breakdown
from final import SELECTIONS, SUBMISSION

from kumite.runs import RunStore

N_BOOTSTRAP = 2000
EXPORTS = COMPETITION / "outputs" / "exports"


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


def compare(base: str, runs: list[str]) -> None:
    store = RunStore(COMPETITION)
    print(f"{base}: CV {store.record(base).oof_score:.5f}")
    for run_id in runs:
        record = store.record(run_id)
        if record.split_hash != store.record(base).split_hash:
            print(f"{run_id}: CV {record.oof_score:.5f}（分割が異なるため bootstrap なし）")
            continue
        point, low, high, share = cluster_bootstrap(base, run_id, store)
        print(
            f"{run_id}: CV {record.oof_score:.5f} | {run_id} - {base}: {point:+.5f} "
            f"95%区間 [{low:+.5f}, {high:+.5f}] 負の割合 {share:.3f}"
        )


def require_selection(plan: str, selections: Path = SELECTIONS) -> None:
    """選定記録が確定していなければ Private 評価を拒否する。"""
    if not (selections / f"{plan}.json").is_file():
        raise ValueError(f"選定計画 {plan} の記録がないため、Private 評価はまだ実行できません")


def private(plan: str, runs: list[str], exports: list[str]) -> None:
    require_selection(plan)
    store = RunStore(COMPETITION)
    export_store = RunStore(EXPORTS)
    targets: list[tuple[str, Path | None]] = [(r, None) for r in runs]
    targets += [(e, export_store.artifact_path(e, SUBMISSION)) for e in exports]
    for name, path in targets:
        with tempfile.TemporaryDirectory() as directory:
            if path is None:
                path = Path(directory) / SUBMISSION
                write_submission(path, store.load_array(name, "test.npy"))
            parts = private_breakdown(path)
        print(f"{name}: Private " + " ".join(f"{k} {v:.5f}" for k, v in parts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("runs", nargs="+")
    p = sub.add_parser("private")
    p.add_argument("--after-selection", required=True, help="確定済みの選定計画名")
    p.add_argument("runs", nargs="*")
    p.add_argument("--export", action="append", default=[])
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.base, args.runs)
    else:
        private(args.after_selection, args.runs, args.export)


if __name__ == "__main__":
    main()
