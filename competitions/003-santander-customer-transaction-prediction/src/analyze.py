"""OOF の予測順位と値の未出現率ごとに誤りを集計する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from features import log_counts
from train import ROOT, data_hash, load_data, make_splits

from kumite.checks import validate_regression
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256
from kumite.runs import RunStore


def analyze(store: RunStore, data: Path, run_id: str) -> pd.DataFrame:
    record = store.record(run_id)
    if record.data_hash != data_hash(data):
        raise ValueError("データが run の記録と一致しません")
    x, y, _, _, _ = load_data(data)
    splits = make_splits(x, y, record.seed)
    if record.split_hash != array_sha256(split_roles(len(y), splits)):
        raise ValueError("分割が run の記録と一致しません")
    pred = store.load_array(run_id, "oof.npy")
    validate_regression(pred, len(y), lower=0.0)
    if (pred > 1).any():
        raise ValueError("OOF 確率が [0, 1] の範囲外です")
    unseen_rate = np.empty(len(y))
    for train, valid in splits:
        _, log_frequency = log_counts(x[train], x[valid])
        unseen_rate[valid] = (log_frequency == 0).mean(axis=1)
    ranks = pd.Series(pred).rank(method="average", pct=True).to_numpy()
    rows = pd.DataFrame(
        {
            "順位帯": np.select(
                [ranks <= 0.1, ranks > 0.9], ["下位10%", "上位10%"], default="中間"
            ),
            "未出現率": pd.cut(unseen_rate, [-np.inf, 0.25, 0.30, 0.35, 0.40, 1.0]),
            "上位帯の陰性数": (ranks > 0.9) & (y == 0),
            "下位帯の陽性数": (ranks <= 0.1) & (y == 1),
        }
    )
    return rows.groupby(["順位帯", "未出現率"], observed=True).agg(
        行数=("上位帯の陰性数", "size"),
        上位帯の陰性数=("上位帯の陰性数", "sum"),
        下位帯の陽性数=("下位帯の陽性数", "sum"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(analyze(RunStore(ROOT), args.data, args.run_id).to_string())
