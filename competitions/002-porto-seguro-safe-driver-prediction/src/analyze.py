"""保存済み予測をハッシュ照合して比較し、順位と確率の違いを調べる。"""

import argparse
import json
from pathlib import Path

import numpy as np
from dataset import COMPETITION_DIR, DATA_FILES, feature_rows, load_data
from metrics import scores
from pipeline import STORE

from kumite.experiments import compare_runs
from kumite.provenance import files_sha256


def analyze(run_ids: list[str]) -> None:
    train, _, _ = load_data()
    y = train["target"].to_numpy()
    base = STORE.record(run_ids[0])
    if base.data_hash != files_sha256(DATA_FILES):
        raise ValueError("現在のデータと run のハッシュが異なります")
    x = feature_rows(train)
    missing = ((x == -1) | x.isna()).sum(axis=1).to_numpy()
    report: dict[str, object] = {}
    for run_id in run_ids:
        record = STORE.record(run_id)
        paired = compare_runs(base, record)
        pred = STORE.load_array(run_id, "oof.npy")
        slices = {}
        for name, mask in (("欠損なし", missing == 0), ("欠損あり", missing > 0)):
            if len(np.unique(y[mask])) == 2:
                slices[name] = {"n": int(mask.sum()), **scores(y[mask], pred[mask])}
        for c in [c for c in x if str(c).endswith("_cat")][:3]:
            for value in x[c].value_counts().head(5).index:
                mask = x[c].to_numpy() == value
                if len(np.unique(y[mask])) == 2:
                    slices[f"{c}={value}"] = {"n": int(mask.sum()), **scores(y[mask], pred[mask])}
        report[run_id] = {
            "scores": scores(y, pred),
            "fold_diffs": paired.diffs,
            "mean_fold_gini": float(np.mean(record.fold_scores)),
            "slices": slices,
            "prediction_quantiles": np.quantile(pred, [0, 0.01, 0.5, 0.99, 1]).tolist(),
            "monotone_example": scores(y, 0.1 + 0.8 * pred),
        }
    out: Path = COMPETITION_DIR / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("-".join(run_ids) + ".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    analyze(parser.parse_args().run_ids)
