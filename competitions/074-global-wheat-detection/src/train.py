"""固定した分割で比較実験を実行し、検出結果を保存する。"""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from dataset import ROOT, load_samples, split_images
from metric import image_score
from model import build_model, detect, fit

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.runs import RunStore

TRIALS = {
    "E00": ("source-v1", 512, False, 0.50, "基準"),
    "E01": ("random-v1", 512, False, 0.50, "同じ撮影元の共有で検証スコアが上がる"),
    "E02": ("source-v1", 768, False, 0.50, "解像度を上げると小さい穂を拾える"),
    "E03": ("source-v1", 512, True, 0.50, "色の拡張で未知の照明条件に汎化する"),
    "E04": ("source-v1", 512, False, 0.65, "NMS を緩めると密集画像の見逃しが減る"),
    "E05": ("source-v1", 512, False, 0.50, "水平反転の重み付き融合で検出が安定する"),
}


def run(trial: str, run_id: str, data: Path, baseline: str | None) -> Path:
    version, size, color, nms, hypothesis = TRIALS[trial]
    if (trial in {"E04", "E05"}) != (baseline is not None):
        raise ValueError("E04 and E05 require --baseline <E00 run ID>")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full CV")
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation, TemporaryDirectory(dir=store.runs_dir) as temp:
        start = perf_counter()
        provenance = code_version(ROOT.parent.parent)
        work = Path(temp)
        samples, excluded = load_samples(data)
        folds = split_images(samples, version)
        split_hash = array_sha256(np.asarray(folds, dtype=np.int64))
        data_hash = files_sha256([data / "train.csv", *(sample.path for sample in samples)])
        weights = [f"fold-{fold}.pt" for fold in range(3)]
        inputs: dict[str, str] = {}
        if baseline is not None:
            base = store.record(baseline)
            if (
                base.params.get("trial") != "E00"
                or base.data_hash != data_hash
                or base.split_version != version
                or base.split_hash != split_hash
            ):
                raise ValueError("Expected E00 checkpoints for the same data and folds")
            inputs = store.input_hashes(baseline, *weights)

        fold_scores: list[float] = []
        all_scores: list[float] = []
        with (work / "oof.jsonl").open("w", encoding="utf-8") as output:
            for fold, weight in enumerate(weights):
                if baseline is None:
                    train = [sample for sample, f in zip(samples, folds, strict=True) if f != fold]
                    model = fit(train, size, color, device)
                    torch.save(model.state_dict(), work / weight)
                else:
                    model = build_model(size, nms, pretrained=False).to(device)
                    checkpoint = store.artifact_path(baseline, weight)
                    model.load_state_dict(
                        torch.load(checkpoint, map_location=device, weights_only=True)
                    )
                scores_in_fold: list[float] = []
                for sample, assignment in zip(samples, folds, strict=True):
                    if assignment != fold:
                        continue
                    boxes, scores = detect(model, sample.path, device, flip_fusion=trial == "E05")
                    truth = np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)
                    score = image_score(truth, boxes, scores)
                    output.write(
                        json.dumps(
                            {
                                "image_id": sample.path.stem,
                                "source": sample.source,
                                "fold": fold,
                                "boxes": boxes.tolist(),
                                "scores": scores.tolist(),
                                "image_score": score,
                            }
                        )
                        + "\n"
                    )
                    if score is not None:
                        scores_in_fold.append(score)
                if not scores_in_fold:
                    raise ValueError(f"Fold {fold} has no scorable images")
                fold_scores.append(float(np.mean(scores_in_fold)))
                all_scores.extend(scores_in_fold)
                print(f"fold {fold}: {fold_scores[-1]:.6f}", flush=True)
                del model

        split = {sample.path.stem: fold for sample, fold in zip(samples, folds, strict=True)}
        (work / "split.json").write_text(json.dumps(split), encoding="utf-8")
        record = ExperimentRecord(
            competition_id="074",
            run_id=run_id,
            metric="wheat-image-precision",
            greater_is_better=True,
            data_version="official-v1",
            data_hash=data_hash,
            split_version=version,
            split_hash=split_hash,
            seed=42,
            hypothesis=hypothesis,
            changed_one_thing=trial,
            model="Faster R-CNN ResNet-50 FPN",
            params={
                "trial": trial,
                "size": size,
                "color": color,
                "nms": nms,
                "flip_fusion": trial == "E05",
                "epochs": 8,
                "batch_size": 2,
                "excluded": excluded,
                "empty_pairs": len(samples) - len(all_scores),
            },
            fold_scores=fold_scores,
            oof_score=float(np.mean(all_scores)),
            runtime_sec=perf_counter() - start,
            code_version=provenance,
            inputs=inputs,
        )

        def write_outputs(output: Path) -> None:
            for path in work.iterdir():
                shutil.copyfile(path, output / path.name)

        return store.publish(reservation, record, write_outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("run_id")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    print(run(args.trial, args.run_id, args.data, args.baseline))


if __name__ == "__main__":
    main()
