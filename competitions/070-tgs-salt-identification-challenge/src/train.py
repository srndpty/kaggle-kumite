"""固定 fold で学習し、評価と提出の証跡を保存する。"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from dataset import Data, data_hash, load_data
from models import BATCH, EPOCHS, fit, load_checkpoint, predict, segment
from segmentation import (
    SEED,
    choose_threshold,
    duplicate_groups,
    make_folds,
    score,
    write_submission,
)

from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "baseline": "BCE・閾値 0.50",
    "threshold": "内側検証で画素閾値を校正",
    "dice": "BCE と soft Dice の等重み平均",
    "flip": "学習時の左右反転",
    "flip-tta": "推論時の左右反転平均",
}


def calibrate(
    data: Data, outer_train: np.ndarray, groups: np.ndarray, device: torch.device
) -> float:
    masks = data.masks[outer_train]
    images = data.train[outer_train]
    inner = make_folds(masks, groups[outer_train], 4)
    model = fit(images[inner != 0], masks[inner != 0], device, "baseline")
    return choose_threshold(masks[inner == 0], predict(model, images[inner == 0], device))


def run(run_id: str, variant: str, baseline: str | None) -> None:
    data_dir = ROOT / "data"
    data = load_data(data_dir)
    groups = duplicate_groups(data.train)
    folds = make_folds(data.masks, groups, 5)
    evaluation = Evaluation(
        "070",
        "mean_iou_precision",
        True,
        score,
        data_hash(data_dir),
        "coverage-group-v1",
        array_sha256(folds),
        SEED,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("全 fold の学習には CUDA が必要です")
    torch.backends.cudnn.deterministic = True
    store = RunStore(ROOT)
    checkpoints = [f"fold-{fold}.pt" for fold in range(5)]
    if baseline:
        record = store.record(baseline)
        if (
            record.params.get("variant") != "baseline"
            or record.data_hash != evaluation.data_hash
            or record.split_hash != evaluation.split_hash
        ):
            raise ValueError("ベースラインの設定・データ・分割が一致しません")
    models: list[dict[str, torch.Tensor]] = []

    def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        fold = len(models)
        print(f"fold {fold}", flush=True)
        if baseline is None:
            model = fit(data.train[train], data.masks[train], device, variant)
        else:
            checkpoint = store.artifact_path(baseline, checkpoints[fold])
            model = load_checkpoint(checkpoint, device)
        models.append({name: value.cpu() for name, value in model.state_dict().items()})
        threshold = calibrate(data, train, groups, device) if variant == "threshold" else 0.5

        return FoldOutput(
            segment(model, data.train[valid], device, threshold, variant),
            segment(model, data.test, device, threshold, variant),
            {"threshold": threshold},
        )

    with store.reserve(run_id) as reservation:
        started = time.monotonic()
        result = cross_fit(folds, fit_fold)

        def write_artifacts(path: Path) -> None:
            np.save(path / "folds.npy", folds)
            np.save(path / "groups.npy", groups)
            for fold, weights in enumerate(models):
                torch.save(weights, path / f"fold-{fold}.pt")
            write_submission(path / "submission.csv", data.test_ids, result.query > 0.5)

        trial = Trial(
            VARIANTS[variant],
            VARIANTS[variant],
            "U-Net",
            "kaggle-v1",
            {
                "variant": variant,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "cuda_dtype": "bfloat16",
            },
            store.input_hashes(baseline, *checkpoints) if baseline else {},
        )
        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            data.masks,
            folds_to_splits(folds),
            result,
            time.monotonic() - started,
            write_artifacts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--baseline", help="校正・推論時拡張で再利用するベースライン run ID")
    args = parser.parse_args()
    if bool(args.baseline) != (args.variant in {"threshold", "flip-tta"}):
        parser.error("threshold と flip-tta の実行時にだけ --baseline を指定してください")
    run(args.run_id, args.variant, args.baseline)


if __name__ == "__main__":
    main()
