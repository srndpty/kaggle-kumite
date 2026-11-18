"""実験の分割・予測・成果物ハッシュと最終固定ファイルを監査する。"""

import json

import numpy as np
import pandas as pd
from dataset import OUTPUT, ROOT, load_prepared, make_folds, submission
from train import accuracy

from kumite.checks import validate_probabilities
from kumite.crossfit import folds_to_splits
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore


def main() -> None:
    store = RunStore(ROOT)
    frame, test, _, _, summary = load_prepared()
    result = {}
    for record in store.records():
        for name in record.artifacts:
            store.artifact_path(record.run_id, name)
        if record.run_id == "P0" or record.params.get("failed"):
            result[record.run_id] = {"artifacts": len(record.artifacts), "cv": False}
            continue
        folds = make_folds(frame, record.seed, record.params["random_split"])
        if record.data_hash != summary["data_hash"] or record.split_hash != array_sha256(folds):
            raise ValueError(f"{record.run_id}: データまたは分割ハッシュ不一致")
        saved = pd.read_csv(store.artifact_path(record.run_id, "manifest.csv"))
        if frame.id.tolist() != saved.id.tolist() or not np.array_equal(folds, saved.fold):
            raise ValueError(f"{record.run_id}: 行順または分割不一致")
        splits = folds_to_splits(folds)
        if record.params["benchmark"]:
            splits = splits[:1]
        valid = np.concatenate([indices for _, indices in splits])
        oof, query = (store.load_array(record.run_id, name) for name in ("oof.npy", "test.npy"))
        validate_probabilities(oof[valid], len(valid), 12)
        validate_probabilities(query, len(test), 12)
        measured = [
            accuracy(frame.label.to_numpy()[indices], oof[indices]) for _, indices in splits
        ]
        if measured != record.fold_scores:
            raise ValueError(f"{record.run_id}: 保存予測からの指標が一致しません")
        pd.testing.assert_frame_equal(
            submission(test, query),
            pd.read_csv(store.artifact_path(record.run_id, "submission.csv")),
        )
        result[record.run_id] = {
            "artifacts": len(record.artifacts),
            "cv": True,
            "folds": len(splits),
        }
    selection = json.loads((OUTPUT / "selection.json").read_text(encoding="utf-8"))
    if selection["submission_sha256"] != file_sha256(
        store.artifact_path(selection["selected"], "submission.csv")
    ):
        raise ValueError("選定済み提出ハッシュ不一致")
    if store.orphans():
        raise ValueError("未確定runがあります")
    result["runtime_seconds"] = sum(r.runtime_sec or 0 for r in store.records())
    result["failed_budget_charge_seconds"] = sum(
        r.params.get("budget_charge_seconds", 0) for r in store.records()
    )
    with (OUTPUT / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
