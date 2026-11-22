"""保存済み E03 の閾値診断と、別実行の提出予測の照合。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import ID, TARGET
from model import classify
from select_model import read_oof, verified_artifact
from train import ROOT

from kumite.experiments import find_record
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def distribution(scores: np.ndarray, thresholds: np.ndarray) -> dict:
    classes = classify(scores, thresholds)
    return {
        "rows": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "quantiles": dict(
            zip(
                ("p05", "p25", "p50", "p75", "p95"),
                np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                strict=True,
            )
        ),
        "class_shares": (np.bincount(classes, minlength=5) / len(scores)).tolist(),
        "thresholds": thresholds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    store = RunStore(ROOT)
    record = find_record(store.records(), "E03")
    oof = read_oof(store, record)
    diagnostics = json.loads(
        verified_artifact(store, record, "diagnostics.json").read_text("utf-8")
    )
    folds = json.loads(verified_artifact(store, record, "fold_info.json").read_text("utf-8"))
    scores = np.load(verified_artifact(store, record, "test_scores.npy"))
    expected = pd.read_csv(verified_artifact(store, record, "submission.csv"), dtype={ID: str})
    actual = pd.read_csv(args.replay / "submission.csv", dtype={ID: str})
    meta = json.loads(args.notebook.read_text("utf-8"))["metadata"]["kumite"]
    replay = json.loads((args.replay / "replay.json").read_text("utf-8"))
    for key in ("source_run", "train_hash", "data_hash", "split_hash", "bundle_sha256"):
        if replay[key] != meta[key]:
            raise ValueError(f"再現検証と Notebook の {key} が一致しません")
    if (replay["source_run"], replay["data_hash"], replay["split_hash"]) != (
        record.run_id,
        record.data_hash,
        record.split_hash,
    ) or meta["source_oof_sha256"] != record.artifacts["oof.csv"]:
        raise ValueError("再現検証の出所が E03 と一致しません")
    if not expected[ID].equals(actual[ID]) or not actual[TARGET].isin(range(5)).all():
        raise ValueError("提出ファイルの PetID または予測値が無効です")
    replay_scores = np.load(args.replay / "test_scores.npy")
    # 連続予測の順序は test.csv、提出の順序は sample_submission.csv に従う。
    if replay_scores.shape != scores.shape or not np.isfinite(replay_scores).all():
        raise ValueError("再現検証の連続予測が無効です")
    output = {
        "validation_run": "E03-replay-v1",
        "notebook_sha256": file_sha256(args.notebook),
        **replay,
        "rows": len(actual),
        "different_classes": int((expected[TARGET] != actual[TARGET]).sum()),
        "max_abs_score_difference": float(np.max(np.abs(scores - replay_scores))),
        "oof_by_fold": {
            str(fold): distribution(
                oof.loc[oof["fold"] == fold, "score"].to_numpy(),
                np.asarray(info["thresholds"]),
            )
            for fold, info in enumerate(folds)
        },
        "pooled_oof_with_submission_thresholds": distribution(
            oof["score"].to_numpy(), np.asarray(diagnostics["submission_thresholds"])
        ),
        "test_fold_mean": distribution(scores, np.asarray(diagnostics["submission_thresholds"])),
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
