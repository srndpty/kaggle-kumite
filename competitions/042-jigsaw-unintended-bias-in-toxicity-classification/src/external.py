"""採用 run を固定した後、公開済みの公式テストラベルで比較する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import FILES, load_train, validate
from metrics import IDENTITIES, breakdown
from model import predict_ensemble
from train import ROOT

from kumite.provenance import array_sha256, file_sha256, files_sha256
from kumite.runs import RunStore


def predict_refit(
    store: RunStore,
    run_ids: tuple[str, ...],
    train: pd.DataFrame,
    query: np.ndarray,
    split_hash: str,
) -> dict[str, np.ndarray]:
    text = train["comment_text"].to_numpy()
    labels = train[["target", *IDENTITIES]].to_numpy() >= 0.5
    target = train["target"].to_numpy()
    predictions = {}
    for run_id in run_ids:
        frame = pd.read_csv(store.artifact_path(run_id, "folds.csv"), dtype={"fold": "int8"})
        folds = frame["fold"].to_numpy()
        if not frame["id"].equals(train["id"]) or array_sha256(folds) != split_hash:
            raise ValueError(f"{run_id} の学習行順または分割が選定記録と一致しません")
        print(f"{run_id}: Public 用推論", flush=True)
        trial = store.record(run_id).params["trial"]
        predictions[run_id] = predict_ensemble(text, labels, target, folds, query, trial, jobs=3)
    return predictions


def predict_saved(
    store: RunStore, run_ids: tuple[str, ...], ids: pd.Index
) -> dict[str, np.ndarray]:
    predictions = {}
    for run_id in run_ids:
        frame = pd.read_csv(store.artifact_path(run_id, "submission.csv")).set_index("id")
        if not (frame.index.is_unique and len(frame) == len(ids) and ids.isin(frame.index).all()):
            raise ValueError(f"{run_id} の提出 ID と公式 Private テストの ID が一致しません")
        predictions[run_id] = frame.loc[ids, "prediction"].to_numpy()
    return predictions


def evaluate(root: Path, split: str) -> dict:
    selection = json.loads((root / "outputs" / "selection.json").read_text(encoding="utf-8"))
    store = RunStore(root)
    selected = store.record(selection["selected"])
    if selected.artifacts["submission.csv"] != selection["submission_hash"]:
        raise ValueError("採用 run の提出ファイルが選定記録と一致しません")
    run_ids = tuple(dict.fromkeys((selection["baseline"], selection["selected"])))
    data = root / "data"
    source = data / f"test_{split}_expanded.csv"
    columns = ["id", "toxicity", *IDENTITIES]
    external = pd.read_csv(source, usecols=columns).set_index("id")
    if not external.index.is_unique or external["toxicity"].isna().any():
        raise ValueError("公式テストに ID の重複またはラベルの欠損があります")
    if split == "public":
        if files_sha256(data / name for name in FILES) != selection["data_hash"]:
            raise ValueError("公式学習データが選定記録と一致しません")
        query = pd.read_csv(source, usecols=["id", "comment_text"], keep_default_na=False)
        if not external.index.equals(pd.Index(query["id"])):
            raise ValueError("Public テストの本文とラベルの順序が一致しません")
        predictions = predict_refit(
            store,
            run_ids,
            load_train(data),
            query["comment_text"].to_numpy(),
            selection["split_hash"],
        )

    else:
        predictions = predict_saved(store, run_ids, external.index)

    labels = external[["toxicity", *IDENTITIES]].to_numpy() >= 0.5
    results = {}
    for run_id, prediction in predictions.items():
        validate(prediction, len(external))
        results[run_id] = breakdown(labels, prediction)
    return {"data_file": source.name, "data_sha256": file_sha256(source), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("public", "private"))
    args = parser.parse_args()
    path = ROOT / "outputs" / f"{args.split}_external.json"
    if path.exists():
        raise FileExistsError(path)
    result = evaluate(ROOT, args.split)
    with path.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2)
    for run_id, metrics in result["results"].items():
        print(f"{run_id}: {metrics['score']:.5f}")
