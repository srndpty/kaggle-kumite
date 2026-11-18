"""保存したfoldモデルから、同じ前処理でtest確率を再生成して照合する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dataset import ROOT, load_prepared, submission
from network import Features, SmallCNN, feature_batches, predict, seed_all

from kumite.runs import RunStore


def reproduce(run_id: str, output: Path | None = None) -> None:
    store = RunStore(ROOT)
    record = store.record(run_id)
    _, test, _, waves, summary = load_prepared(deep=True)
    if record.data_hash != summary["data_hash"]:
        raise ValueError("データハッシュ不一致")
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = feature_batches(
        waves, Features(record.params["experiment"] == "E02").to(device), device
    )
    probabilities = []
    for k in range(3):
        checkpoint = torch.load(
            store.artifact_path(run_id, f"fold{k}.pt"), map_location=device, weights_only=True
        )
        model = SmallCNN(checkpoint["state_dict"]["layers.14.weight"].shape[0]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities.append(predict(model, features, checkpoint["mean"], checkpoint["std"]))
    result = np.mean(np.stack(probabilities).astype(np.float64), axis=0)
    saved = store.load_array(run_id, "test.npy")
    maximum = float(np.abs(saved - result).max())
    if not np.allclose(saved, result, rtol=1e-5, atol=1e-6):
        raise ValueError(f"再生成予測が不一致: 最大差{maximum}")
    if not submission(test, saved).equals(submission(test, result)):
        raise ValueError("再生成した提出ラベルが一致しません")
    path = output if output is not None else ROOT / "outputs" / f"{run_id}-reproduction.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            dict(run_id=run_id, max_probability_difference=maximum, labels_equal=True), stream
        )
    print(f"再生成の最大確率差: {maximum}; 提出ラベル一致")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reproduce(args.run_id, args.output)
