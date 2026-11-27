"""公開済み OOF のセッション対応差と入力長別の評価。"""

import argparse
import csv
import gzip
import json
from collections import Counter
from itertools import islice

import numpy as np
from dataset import ROOT, sessions
from evaluation import ACTIONS, SEED, WEIGHTS, split_session
from workflow import Context

from kumite.runs import RunStore

BOOTSTRAPS = 1000


def paired_bootstrap(windows: list[Counter[tuple[int, ...]]]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    totals = np.zeros((BOOTSTRAPS, 6), dtype=np.int64)
    for window in windows:
        values = np.array(list(window), dtype=np.int64)
        frequencies = np.array(list(window.values()), dtype=np.int64)
        count = int(frequencies.sum())
        if count == 0:
            raise ValueError("対応差を計算できない空の検証窓です")
        # 同じ分子差・分母を持つセッションの再標本化を多項分布にまとめる。
        weights = rng.multinomial(count, frequencies / count, size=BOOTSTRAPS)
        totals += weights @ values
    if np.any(totals[:, 3:] == 0):
        raise ValueError("再標本化した少なくとも1行動の分母が0です")
    scores = (totals[:, :3] / totals[:, 3:]) @ np.asarray(WEIGHTS)
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def compare(old_run: str, new_run: str) -> dict[str, object]:
    context = Context.load(ROOT / "selection-p4.toml")
    store = RunStore(ROOT)
    old = store.record(old_run)
    new = store.record(new_run)
    if old.split_hash != new.split_hash or old.data_hash != new.data_hash:
        raise ValueError("比較する run の正本データまたは分割が異なります")
    paired: list[Counter[tuple[int, ...]]] = []
    by_length: dict[str, Counter[str]] = {}
    with (
        gzip.open(store.artifact_path(old_run, "oof.csv.gz"), "rt", encoding="utf-8") as old_file,
        gzip.open(store.artifact_path(new_run, "oof.csv.gz"), "rt", encoding="utf-8") as new_file,
    ):
        old_rows = csv.reader(old_file)
        new_rows = csv.reader(new_file)
        for fold, window in enumerate(context.windows):
            pair_counts: Counter[tuple[int, ...]] = Counter()
            for session, events in sessions(window):
                prefix, truth = split_session(session, events)
                length = len(prefix)
                bucket = "1" if length == 1 else "2–4" if length <= 4 else "5+"
                counts = by_length.setdefault(bucket, Counter())
                delta: list[int] = []
                denoms: list[int] = []
                for action, a, b in zip(
                    ACTIONS, islice(old_rows, 3), islice(new_rows, 3), strict=True
                ):
                    expected = [str(fold), str(session), action]
                    if a[:3] != expected or b[:3] != expected:
                        raise ValueError("比較する OOF のキーが一致しません")
                    targets = truth[action]
                    denominator = min(20, len(targets))
                    old_hits = len(targets.intersection(map(int, a[3].split())))
                    new_hits = len(targets.intersection(map(int, b[3].split())))
                    delta.append(new_hits - old_hits)
                    denoms.append(denominator)
                    counts[f"{action}_old_hits"] += old_hits
                    counts[f"{action}_new_hits"] += new_hits
                    counts[f"{action}_denominator"] += denominator
                pair_counts[tuple(delta + denoms)] += 1
            paired.append(pair_counts)
        if next(old_rows, None) is not None or next(new_rows, None) is not None:
            raise ValueError("比較する OOF に余分なキーがあります")
    return {
        "score_difference": new.oof_score - old.oof_score,
        "bootstrap_95_percent": paired_bootstrap(paired),
        "prefix_length": by_length,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_run")
    parser.add_argument("new_run")
    args = parser.parse_args()
    print(json.dumps(compare(args.old_run, args.new_run), ensure_ascii=False, indent=2))
