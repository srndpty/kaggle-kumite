"""固定分割で音声タグ付けモデルを学習し、OOF と重みを保存する。"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataset import (
    FOLDS,
    SEED,
    data_fingerprint,
    eligible_noisy,
    read_clips,
    read_wave,
    split_groups,
    targets,
)
from model import Tagger, fit, predict

from kumite.crossfit import CrossFitResult
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.lwlrap import lwlrap
from kumite.provenance import array_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "E00": ("タグ頻度", "定数予測の基準"),
    "E01": ("log-mel CNN", "音響特徴によってタグ順位が改善する"),
    "E02": ("noisy 25%", "希少タグの順位が改善する"),
    "E03": ("最大値集約", "短いイベントの順位が改善する"),
    "E04": ("時間マスク", "時間帯への依存が減る"),
    "E05": ("noisy で先に学習", "curated で仕上げると同時混合よりタグ順位が改善する"),
}


def run(
    run_id: str,
    experiment: str,
    updates: int,
    benchmark: bool,
    seconds: float,
    baseline: str | None,
) -> Path:
    start = time.monotonic()
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        root = ROOT / "data"
        sample = pd.read_csv(root / "sample_submission.csv", nrows=0)
        tags = list(sample.columns[1:])
        curated = read_clips(root, "curated")
        noisy = read_clips(root, "noisy")
        labels = targets(curated, tags)
        folds = split_groups(curated, labels)
        manifest = pd.DataFrame(
            {
                "fname": [clip.path.name for clip in curated],
                "group": [c.group for c in curated],
                "seconds": [c.seconds for c in curated],
                "fold": folds,
            }
        )
        data_hash = data_fingerprint(root, curated + noisy)
        split_hash = array_sha256(folds)
        split_version = "curated-group-3fold-v1" + ("-benchmark" if benchmark else "")
        inputs: dict[str, str] = {}
        if baseline is not None:
            record = store.record(baseline)
            if (
                record.params.get("experiment") != "E01"
                or record.params.get("updates") != updates
                or record.data_hash != data_hash
                or record.split_hash != split_hash
                or record.split_version != split_version
            ):
                raise ValueError("Baseline must be the matching E01 run")
            inputs = store.input_hashes(baseline, *(f"fold{i}.pt" for i in range(FOLDS)))
        predictions = np.full(labels.shape, np.nan)
        splits = []
        infos = []
        checkpoints = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waves = [read_wave(clip.path) for clip in curated] if experiment != "E00" else []
        for fold in range(1 if benchmark else FOLDS):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            fitting = [curated[i] for i in train]
            validation = [curated[i] for i in valid]
            assert not {clip.group for clip in fitting} & {clip.group for clip in validation}
            allowed = eligible_noisy(noisy, validation) if experiment in {"E02", "E05"} else []
            splits.append((train, valid))
            infos.append({"fold": fold, "eligible_noisy": len(allowed)})
            if experiment == "E00":
                predictions[valid] = labels[train].mean(axis=0)
                continue
            if baseline is not None:
                checkpoint = store.artifact_path(baseline, f"fold{fold}.pt")
                model = Tagger(len(tags)).to(device)
                model.load_state_dict(
                    torch.load(checkpoint, map_location=device, weights_only=True)
                )
                model.eval()
            else:
                model = fit(
                    [waves[i] for i in train],
                    labels[train],
                    [c.path for c in allowed],
                    targets(allowed, tags),
                    updates=updates,
                    seed=SEED + fold,
                    device=device,
                    deadline=start + seconds,
                    time_mask=experiment == "E04",
                    staged_noisy=experiment == "E05",
                )
            for i in valid:
                predictions[i] = predict(model, waves[i], device, maximum=experiment == "E03")
            checkpoints.append({name: value.cpu() for name, value in model.state_dict().items()})
        evaluated = predictions[np.concatenate([valid for _, valid in splits])]
        if not (np.isfinite(evaluated).all() and ((evaluated >= 0) & (evaluated <= 1)).all()):
            raise ValueError("Invalid OOF scores")
        evaluation = Evaluation(
            ROOT.name,
            "lwlrap",
            True,
            lwlrap,
            data_hash,
            split_version,
            split_hash,
            SEED,
        )
        change, hypothesis = TRIALS[experiment]
        trial = Trial(
            hypothesis,
            change,
            "frequency" if experiment == "E00" else "SmallCNN",
            "official-v1",
            {
                "updates": updates,
                "experiment": experiment,
                "tags": tags,
                "device": str(device),
                "benchmark": benchmark,
                "noisy_warmup_updates": updates // 4 if experiment == "E05" else 0,
            },
            inputs,
        )

        def save(path: Path) -> None:
            manifest.to_csv(path / "manifest.csv", index=False)
            np.save(path / "labels.npy", labels)
            for fold, checkpoint in enumerate(checkpoints):
                torch.save(checkpoint, path / f"fold{fold}.pt")

        result = CrossFitResult(predictions, np.empty((0, len(tags))), infos)
        return publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            labels,
            splits,
            result,
            time.monotonic() - start,
            save,
            save_query=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--experiment", choices=TRIALS, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    if args.updates <= 0 or args.seconds <= 0:
        parser.error("updates and seconds must be positive")
    if (args.experiment == "E03") != (args.baseline is not None):
        parser.error("--baseline is required for E03 only")
    if args.baseline is not None and args.benchmark:
        parser.error("E03 uses the full E01 run")
    run(args.run_id, args.experiment, args.updates, args.benchmark, args.seconds, args.baseline)
