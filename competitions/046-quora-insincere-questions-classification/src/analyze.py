"""固定した OOF 予測のペア比較と誤りの集計。"""

import argparse
import json

import numpy as np
import pandas as pd
from model import SEED, group_keys
from train import ROOT, read_inputs

from kumite.provenance import files_sha256
from kumite.runs import RunStore


def paired_interval(
    y: np.ndarray, base: np.ndarray, other: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    values = np.column_stack((y * base, y + base, y * other, y + other))
    grouped = pd.DataFrame(values).groupby(groups, sort=False).sum().to_numpy()
    signatures, counts = np.unique(grouped, axis=0, return_counts=True)
    # 同じ計数を持つ文面グループをまとめ、復元抽出と等価な多項分布で引く。
    draws = np.random.default_rng(SEED).multinomial(len(grouped), counts / len(grouped), 2000)
    tp_base, denom_base, tp_other, denom_other = (draws @ signatures).T
    low, high = np.quantile(
        2 * tp_other / np.maximum(1, denom_other) - 2 * tp_base / np.maximum(1, denom_base),
        [0.025, 0.975],
    )
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("other")
    args = parser.parse_args()
    data = ROOT / "data"
    train, _ = read_inputs(data)
    y = train["target"].to_numpy()
    groups = group_keys(train["question_text"])
    store = RunStore(ROOT)
    base = np.load(store.artifact_path(args.base, "oof.npy"))
    other = np.load(store.artifact_path(args.other, "oof.npy"))
    with (
        np.load(store.artifact_path(args.base, "probabilities.npz")) as a,
        np.load(store.artifact_path(args.other, "probabilities.npz")) as b,
    ):
        if any(a[key] != b[key] for key in ("data_hash", "split_hash")):
            raise ValueError("データまたは分割が一致しません")
        if str(a["data_hash"]) != files_sha256([data / "train.csv", data / "test.csv"]):
            raise ValueError("現在の CSV と計測時のデータが一致しません")
        if (args.base, args.other) == ("E00", "E01"):
            np.testing.assert_array_equal(a["oof"], b["oof"])
        coverage = b["coverage"]
    print(f"F1 差の 95% 区間: {paired_interval(y, base, other, groups)}")
    length = train["question_text"].str.len().to_numpy()
    for name, feature, bins in (
        ("length", length, np.quantile(length, [0.25, 0.5, 0.75])),
        ("coverage", coverage, np.array([0.5, 0.9, 1.0])),
    ):
        assignments = np.searchsorted(bins, feature, side="left")
        if name == "coverage":
            assignments[feature == 1] = 3
        for bucket in range(4):
            rows = np.flatnonzero(assignments == bucket)
            print(
                json.dumps(
                    {
                        "feature": name,
                        "bucket": bucket + 1,
                        "rows": len(rows),
                        "cutoffs": bins.tolist(),
                        "positives": int(y[rows].sum()),
                        "base_fp": int(np.count_nonzero((base[rows] == 1) & (y[rows] == 0))),
                        "base_fn": int(np.count_nonzero((base[rows] == 0) & (y[rows] == 1))),
                        "fp": int(np.count_nonzero((other[rows] == 1) & (y[rows] == 0))),
                        "fn": int(np.count_nonzero((other[rows] == 0) & (y[rows] == 1))),
                    }
                )
            )


if __name__ == "__main__":
    main()
