"""同条件runの比較と、固定した区分別のエラー分析。"""

import argparse
import json
from typing import Any

import numpy as np
from cv import text_key
from dataset import COMPETITION_DIR, DATA_FILES, LABELS, load_data
from metrics import scores
from pipeline import STORE

from kumite.experiments import compare_runs
from kumite.provenance import files_sha256


def compare(run_ids):
    records = [STORE.record(r) for r in run_ids]
    base = records[0]
    for r in records:
        compare_runs(base, r)
    return [
        {
            "run_id": r.run_id,
            "mean_fold_auc": float(np.mean(r.fold_scores)),
            "oof_auc": r.oof_score,
            "delta": float(np.mean(r.fold_scores) - np.mean(base.fold_scores)),
            "runtime_sec": r.runtime_sec,
        }
        for r in records
    ]


def errors(run_id):
    train, test, _ = load_data()
    if STORE.record(run_id).data_hash != files_sha256(DATA_FILES):
        raise ValueError("現在のデータがrunと異なります")
    pred = STORE.load_array(run_id, "oof.npy")
    y = train[list(LABELS)].to_numpy()
    lengths = train.comment_text.str.len().to_numpy()
    slices = {
        "short_le_100": lengths <= 100,
        "medium": (lengths > 100) & (lengths <= 1000),
        "long_gt_1000": lengths > 1000,
        "non_ascii": train.comment_text.map(lambda t: not str(t).isascii()).to_numpy(),
        "train_test_shared_text": train.comment_text.map(text_key)
        .isin(set(test.comment_text.map(text_key)))
        .to_numpy(),
    }
    report: dict[str, Any] = {"run_id": run_id, "overall": scores(y, pred), "slices": {}}
    for name, mask in slices.items():
        entry: dict[str, Any] = {"rows": int(mask.sum())}
        if mask.any() and ((y[mask].sum(0) > 0) & (y[mask].sum(0) < mask.sum())).all():
            entry["metrics"] = scores(y[mask], pred[mask])
        report["slices"][name] = entry
    loss = -(
        y * np.log(np.clip(pred, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-15, 1))
    ).mean(1)
    report["largest_error_ids"] = train.id.iloc[np.argsort(loss)[-30:][::-1]].tolist()
    out = COMPETITION_DIR / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--errors", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(errors(a.runs[0]) if a.errors else compare(a.runs), ensure_ascii=False, indent=2)
    )
