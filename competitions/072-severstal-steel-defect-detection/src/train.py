"""公式 Dice で固定3 fold の比較実験を実行する。"""

import argparse
import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from cv_checks import read_sources, source_hashes, verify_sources
from model import SteelUNet, make_presence_model, normalize, training_loss
from PIL import Image
from split import SEED, SPLIT_VERSION, make_folds
from steel import CLASSES, SHAPE, decode_rle, dice, load_labels
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
BATCH = 8
EPOCHS = 10
LR = 3e-4
WEIGHTS = "resnet18-f37072fd.pth"
VARIANTS = {
    "baseline": "BCE の基準",
    "dice": "非空マスク向けの soft Dice loss を併用",
    "full": "入力解像度を 256×1600 に変更",
    "area": "32画素未満のクラス予測を空にする",
    "gate": "独立した欠陥有無分類器でクラス予測を空にする",
}


class SteelImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        directory: Path,
        labels: dict[str, dict[int, str]],
        ids: list[str],
        shape: tuple[int, int],
        *,
        resize_masks: bool,
        presence: bool = False,
    ) -> None:
        self.directory = directory
        self.labels = labels
        self.ids = ids
        self.shape = shape
        self.resize_masks = resize_masks
        self.presence = presence

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        height, width = self.shape
        with Image.open(self.directory / image_id) as image:
            pixels = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.array(pixels)).permute(2, 0, 1)
        if self.presence:
            return x, torch.tensor([class_id in self.labels[image_id] for class_id in CLASSES])
        masks = np.stack(
            [decode_rle(self.labels[image_id].get(class_id, "")) for class_id in CLASSES]
        )
        if self.resize_masks and self.shape != SHAPE:
            masks = masks[:, ::2, ::2].copy()
        return x, torch.from_numpy(masks)


def fit(dataset: SteelImages, weights: Path, variant: str, device: torch.device) -> SteelUNet:
    torch.manual_seed(SEED)
    return optimize(SteelUNet(weights), dataset, device, use_dice=variant == "dice")


def fit_presence(dataset: SteelImages, weights: Path, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    return optimize(make_presence_model(weights), dataset, device, use_dice=False)


def optimize[Model: nn.Module](
    model: Model, dataset: SteelImages, device: torch.device, *, use_dice: bool
) -> Model:
    model.to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    for epoch in range(EPOCHS):
        for images, target in loader:
            x = normalize(images, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = training_loss(
                logits.float(), target.to(device, dtype=torch.float32), use_dice=use_dice
            )
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{EPOCHS}", flush=True)
    return model


@torch.inference_mode()
def predict(
    model: SteelUNet, images: torch.Tensor, device: torch.device, gate: nn.Module | None = None
) -> np.ndarray:
    x = normalize(images, device)
    probability = model(x).float().sigmoid()
    probability = F.interpolate(probability, size=SHAPE, mode="bilinear", align_corners=False)
    values = probability.cpu().numpy()
    if values.shape != (len(images), len(CLASSES), *SHAPE) or not np.isfinite(values).all():
        raise ValueError("予測の形状または有限性が不正です")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("予測確率が範囲外です")
    masks = values >= 0.5
    if gate is not None:
        presence = gate(x).float().sigmoid().cpu().numpy()
        if presence.shape != (len(images), len(CLASSES)) or not np.isfinite(presence).all():
            raise ValueError("欠陥有無の予測が不正です")
        masks &= presence[:, :, None, None] >= 0.5
    return masks


def training_data_hash(data: Path, image_ids: Iterable[str]) -> str:
    return files_sha256(
        [
            data / "train.csv",
            data / WEIGHTS,
            *(data / "train_images" / name for name in sorted(image_ids)),
        ]
    )


def run(run_id: str, variant: str, baseline: str | None) -> None:
    hashes = source_hashes(read_sources())
    data = ROOT / "data"
    directory = data / "train_images"
    weights = data / WEIGHTS
    store = RunStore(ROOT)
    device = torch.device("cuda")
    shape = SHAPE if variant == "full" else (128, 800)
    labels = load_labels(data)
    ids = list(labels)
    folds = make_folds(data, labels)
    split_hash = array_sha256(np.array([folds[image_id] for image_id in ids]))
    data_hash = training_data_hash(data, ids)
    if baseline is not None:
        source = store.record(baseline)
        verify_sources(source, hashes)
        if (source.params.get("variant"), source.split_hash, source.data_hash) != (
            "baseline",
            split_hash,
            data_hash,
        ):
            raise ValueError("基準 run の設定、データまたは分割が一致しません")
    scores: list[float] = []
    rows: list[tuple[str, int, int, float, int, int, bool]] = []
    checkpoints: list[dict[str, torch.Tensor]] = []
    gate_checkpoints: list[dict[str, torch.Tensor]] = []
    input_hashes = (
        store.input_hashes(baseline, *(f"fold-{i}.pt" for i in range(3))) if baseline else {}
    )
    with store.reserve(run_id) as reservation:
        started = time.monotonic()
        source_version = code_version(REPO_ROOT)
        for fold in range(3):
            train = [image_id for image_id in ids if folds[image_id] != fold]
            valid = [image_id for image_id in ids if folds[image_id] == fold]
            if baseline is None:
                dataset = SteelImages(directory, labels, train, shape, resize_masks=True)
                model = fit(dataset, weights, variant, device)
            else:
                model = SteelUNet().to(device).to(memory_format=torch.channels_last)
                checkpoint = store.artifact_path(baseline, f"fold-{fold}.pt")
                model.load_state_dict(
                    torch.load(checkpoint, map_location=device, weights_only=True)
                )
            checkpoints.append({key: tensor.cpu() for key, tensor in model.state_dict().items()})
            model.eval()
            gate = None
            if variant == "gate":
                dataset = SteelImages(
                    directory, labels, train, shape, resize_masks=False, presence=True
                )
                gate = fit_presence(dataset, weights, device).eval()
                gate_checkpoints.append(
                    {key: tensor.cpu() for key, tensor in gate.state_dict().items()}
                )
            limits = {}
            for class_id in CLASSES:
                areas = [
                    sum(map(int, labels[image_id][class_id].split()[1::2]))
                    for image_id in train
                    if class_id in labels[image_id]
                ]
                limits[class_id] = float(np.quantile(areas, 0.25))
            dataset = SteelImages(directory, labels, valid, shape, resize_masks=False)
            first = len(rows)
            loader = DataLoader(dataset, batch_size=4, num_workers=2)
            for batch, (images, targets) in enumerate(loader):
                masks = predict(model, images, device, gate)
                if variant == "area":
                    masks &= masks.sum(axis=(-2, -1), keepdims=True) >= 32
                for index, (truth, prediction) in enumerate(
                    zip(targets.numpy(), masks, strict=True)
                ):
                    image_id = valid[batch * 4 + index]
                    for class_id, target, mask in zip(CLASSES, truth, prediction, strict=True):
                        area = int(target.sum())
                        rows.append(
                            (
                                image_id,
                                class_id,
                                fold,
                                dice(target, mask),
                                area,
                                int(mask.sum()),
                                0 < area <= limits[class_id],
                            )
                        )
            scores.append(float(np.mean([row[3] for row in rows[first:]])))
            print(f"fold {fold}: {scores[-1]:.6f}", flush=True)

        def write_artifacts(output: Path) -> None:
            for fold, checkpoint in enumerate(checkpoints):
                torch.save(checkpoint, output / f"fold-{fold}.pt")
            for fold, checkpoint in enumerate(gate_checkpoints):
                torch.save(checkpoint, output / f"presence-{fold}.pt")
            with (output / "oof.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "image_id",
                        "class_id",
                        "fold",
                        "dice",
                        "target_area",
                        "predicted_area",
                        "small",
                    )
                )
                writer.writerows(rows)
            summary = {
                "all_empty_dice": sum(row[4] == 0 for row in rows) / len(rows),
                "class_dice": {
                    class_id: float(np.mean([row[3] for row in rows if row[1] == class_id]))
                    for class_id in CLASSES
                },
                "positive_dice": float(np.mean([row[3] for row in rows if row[4] > 0])),
                "empty_false_positive_rate": sum(row[4] == 0 and row[5] > 0 for row in rows)
                / sum(row[4] == 0 for row in rows),
                "small_miss_rate": sum(row[6] and row[5] == 0 for row in rows)
                / sum(row[6] for row in rows),
                "small_dice": float(np.mean([row[3] for row in rows if row[6]])),
            }
            (output / "diagnostics.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

        score = float(np.mean([row[3] for row in rows]))
        if source_hashes(read_sources()) != hashes:
            raise ValueError("CV 実行中にソースが変更されました")
        record = ExperimentRecord(
            competition_id="072",
            run_id=run_id,
            metric="mean_dice",
            greater_is_better=True,
            data_version="kaggle-v1",
            data_hash=data_hash,
            split_version=SPLIT_VERSION,
            split_hash=split_hash,
            seed=SEED,
            hypothesis=VARIANTS[variant],
            changed_one_thing=VARIANTS[variant],
            model="ResNet18 U-Net",
            params={
                "variant": variant,
                "shape": shape,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "learning_rate": LR,
            },
            fold_scores=scores,
            oof_score=score,
            runtime_sec=time.monotonic() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=source_version,
            inputs={**input_hashes, **hashes},
        )
        store.publish(reservation, record, write_artifacts)
        print(f"OOF mean Dice: {score:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--baseline", help="面積ゲートに再利用する E00 の run ID")
    args = parser.parse_args()
    if bool(args.baseline) != (args.variant in {"area", "gate"}):
        parser.error("area と gate の run だけ --baseline が必要です")
    run(args.run_id, args.variant, args.baseline)


if __name__ == "__main__":
    main()
