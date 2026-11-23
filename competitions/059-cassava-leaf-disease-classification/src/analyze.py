"""固定分割と OOF のクラス別誤りを集計する。"""

import argparse

import numpy as np
import pandas as pd
from dataset import OUTPUT, ROOT, SPLIT_VERSION
from sklearn.metrics import confusion_matrix

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_probabilities
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    rows = pd.read_csv(OUTPUT / f"{SPLIT_VERSION}.csv")
    group_sizes = rows.groupby("group").size()
    mixed = int((rows.groupby("group")["label"].nunique() > 1).sum())
    print(
        f"{len(rows)} 画像、{len(group_sizes)} グループ、複数画像 {int((group_sizes > 1).sum())}、"
        f"最大 {int(group_sizes.max())} 画像、ラベル不一致 {mixed} グループ"
    )
    print(rows[["width", "height"]].describe().to_string())
    print(pd.crosstab(rows["fold"], rows["label"]).to_string())
    labels = rows["label"].to_numpy()
    store = RunStore(ROOT)
    baseline: np.ndarray | None = None
    for run in args.runs:
        probability = store.load_array(run, "oof.npy")
        validate_probabilities(probability, len(rows), 5)
        pred = probability.argmax(axis=1)
        correct = (pred == labels).astype(float)
        matrix = confusion_matrix(labels, pred, labels=list(range(5)))
        print(run, float(correct.mean()), "クラス別再現率", matrix.diagonal() / matrix.sum(axis=1))
        print(matrix)
        if baseline is None:
            baseline = correct
        else:
            interval = group_bootstrap_diff(
                baseline, correct, rows["group"].to_numpy(), transform=lambda values: values
            )
            print(f"差 {interval.point:+.6f}、95% 区間 [{interval.low:+.6f}, {interval.high:+.6f}]")


if __name__ == "__main__":
    main()
