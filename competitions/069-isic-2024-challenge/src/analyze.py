"""保存済み OOF の飽和と患者当たり病変数別の誤りを集計する。"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from validation import pauc
from workflow import ROOT, TRAIN

from kumite.provenance import file_sha256
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    frame = pd.read_csv(TRAIN, usecols=["target", "patient_id"])
    y = frame["target"].to_numpy()
    counts = frame.groupby("patient_id")["target"].transform("size").to_numpy()
    store = RunStore(ROOT)
    data_hash = file_sha256(TRAIN)
    for run_id in args.run_ids:
        if store.record(run_id).data_hash != data_hash:
            raise ValueError(f"{run_id}: 評価データが学習時と一致しません")
        p = store.load_array(run_id, "oof.npy")
        _, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
        threshold = thresholds[np.flatnonzero(tpr >= 0.8)[0]]
        print(
            f"{run_id}: pAUC={pauc(y, p):.6f} "
            f"AUC={roc_auc_score(y, p):.6f} log_loss={log_loss(y, p):.6f} "
            f"p=0: {(p == 0).sum()}, p=1: {(p == 1).sum()}, threshold={threshold:.8g}"
        )
        for lower, upper in ((1, 115), (116, 241), (242, 477), (478, int(counts.max()))):
            rows = (counts >= lower) & (counts <= upper)
            labels, scores = y[rows], p[rows]
            positive = labels == 1
            called = scores >= threshold
            print(
                f"  {lower}-{upper}: n={len(scores)}, positive={positive.sum()}, "
                f"FN={(positive & ~called).sum()}, FP={(~positive & called).sum()}"
            )


if __name__ == "__main__":
    main()
