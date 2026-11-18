"""ハッシュ照合済みOOFの指標と顧客群ごとの誤差を比較する。"""

import argparse
import json

import numpy as np
from dataset import COMPETITION_DIR, DATA_FILES, KEY, load_data
from features import application_features
from pipeline import STORE, auc, load_history
from sklearn.metrics import log_loss

from kumite.experiments import compare_runs
from kumite.provenance import files_sha256


def analyze(run_ids: list[str]) -> dict[str, object]:
    import pandas as pd

    train, _, _ = load_data()
    y = train["TARGET"].to_numpy()
    groups: dict[str, np.ndarray] = {}
    for name in ("bureau", "previous", "installments"):
        history = load_history(name)
        counts = history.set_index(KEY)[f"{name}_count"]
        n = train[KEY].map(counts).fillna(0).to_numpy()
        for label, mask in (("none", n == 0), ("1-10", (n > 0) & (n <= 10)), ("11+", n > 10)):
            groups[f"{name}:{label}"] = mask
    missing = application_features(train).isna().sum(axis=1).to_numpy()
    groups["missing:0-20"] = missing <= 20
    groups["missing:21+"] = missing > 20
    result: dict[str, object] = {}
    base = STORE.record(run_ids[0])
    data_hash = files_sha256(DATA_FILES)
    for run_id in run_ids:
        record = STORE.record(run_id)
        compare_runs(base, record)
        if record.data_hash != data_hash or record.params.get("benchmark"):
            raise ValueError("現在のデータと異なるrun、または試走です")
        folds = pd.read_csv(STORE.artifact_path(run_id, "folds.csv"))
        if not np.array_equal(folds[KEY], train[KEY]):
            raise ValueError("OOFのID順序が一致しません")
        pred = STORE.load_array(run_id, "oof.npy")
        sections = {}
        for name, mask in groups.items():
            sections[name] = {
                "rows": int(mask.sum()),
                "auc": auc(y[mask], pred[mask]) if len(np.unique(y[mask])) == 2 else None,
                "logloss": float(log_loss(y[mask], pred[mask], labels=[0, 1]))
                if mask.any()
                else None,
            }
        worst = np.argsort(
            -(y * np.log(np.clip(pred, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-15, 1)))
        )[-30:][::-1]
        result[run_id] = {
            "mean_fold_auc": float(np.mean(record.fold_scores)),
            "oof_auc": record.oof_score,
            "fold_differences": compare_runs(base, record).diffs,
            "groups": sections,
            "worst_ids": train.iloc[worst][KEY].tolist(),
        }
    output = COMPETITION_DIR / "outputs" / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_ids), ensure_ascii=False, indent=2))
