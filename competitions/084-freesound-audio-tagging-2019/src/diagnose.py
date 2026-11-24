"""保存済み E05 の3 fold で公開音源を推論し、過去の提出と比較する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from dataset import read_wave
from model import Tagger, predict
from train import ROOT

from kumite.provenance import file_sha256
from kumite.runs import RunStore


def compare(
    template: pd.DataFrame,
    predictions: dict[str, Path],
    checkpoint_hashes: dict[str, str],
) -> dict[str, object]:
    arrays = {}
    for label, path in predictions.items():
        frame = pd.read_csv(path)
        if not frame.columns.equals(template.columns) or not frame["fname"].equals(
            template["fname"]
        ):
            raise ValueError(f"Submission manifest mismatch: {label}")
        scores = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid scores: {label}")
        arrays[label] = scores
    reference = arrays["E05-fold"]
    tags = list(template.columns[1:])
    # 同点は公式列順で扱う。
    reference_ranks = np.argsort(np.argsort(reference, axis=1, kind="stable"), axis=1)

    def summarize(scores: npt.NDArray[np.float64], path: Path) -> dict[str, object]:
        ranks = np.argsort(np.argsort(scores, axis=1, kind="stable"), axis=1)
        top = scores.argmax(axis=1)
        counts = np.bincount(top)
        correlation = 1 - 6 * ((ranks - reference_ranks) ** 2).sum(axis=1) / (
            len(tags) * (len(tags) ** 2 - 1)
        )
        return {
            "sha256": file_sha256(path),
            "top_tag_agreement_with_e05_fold": float((top == reference.argmax(axis=1)).mean()),
            "mean_rank_correlation_with_e05_fold": float(correlation.mean()),
            "unique_top_tags": len(set(top)),
            "most_frequent_top_tag": tags[int(counts.argmax())],
            "most_frequent_top_count": int(counts.max()),
            "mean_score": float(scores.mean()),
            "mean_max_score": float(scores.max(axis=1).mean()),
        }

    return {
        "public_rows": len(template),
        "checkpoint_sha256": checkpoint_hashes,
        **{label: summarize(arrays[label], path) for label, path in predictions.items()},
    }


def diagnose(baseline: Path, final: Path, output: Path) -> None:
    template = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    tags = list(template.columns[1:])
    torch.set_num_threads(4)
    device = torch.device("cpu")
    store = RunStore(ROOT)
    checkpoints = {f"fold{i}.pt": store.artifact_path("E05", f"fold{i}.pt") for i in range(3)}
    models = []
    for checkpoint in checkpoints.values():
        model = Tagger(len(tags)).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        models.append(model)
    rows = []
    for name in template["fname"]:
        wave = read_wave(ROOT / "data" / name)
        rows.append(
            np.mean([predict(model, wave, device, maximum=False) for model in models], axis=0)
        )
    values = np.stack(rows)
    if values.shape != (len(template), len(tags)) or not np.isfinite(values).all():
        raise ValueError("Invalid E05-fold prediction shape or finite values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid E05-fold score range")
    output.mkdir(parents=True, exist_ok=False)
    csv = output / "submission.csv"
    table = pd.DataFrame(values, columns=tags)
    table.insert(0, "fname", template["fname"])
    table.to_csv(csv, index=False)
    result = compare(
        template,
        {"E01": baseline, "final-E05": final, "E05-fold": csv},
        {name: file_sha256(path) for name, path in checkpoints.items()},
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnose(args.baseline, args.final, args.output)
