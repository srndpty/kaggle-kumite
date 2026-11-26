"""Mask R-CNN の学習と画像単位推論。"""

from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from data import SEED, Image
from masks import encode, pixels
from PIL import Image as PillowImage
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


class Settings(TypedDict):
    rank_by_mask: NotRequired[bool]
    min_size: int
    max_size: int
    flip: float
    retain: float
    mask_threshold: float
    detections: int
    score_threshold: float
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float


def settings(trial: str) -> Settings:
    return Settings(
        min_size=800 if trial == "E01" else 640,
        max_size=1333 if trial == "E01" else 1024,
        flip=0.5 if trial == "E02" else 0.0,
        retain=0.5 if trial == "E03" else 0.0,
        mask_threshold=0.5,
        detections=800,
        score_threshold=0.2,
        epochs=12,
        batch_size=2,
        learning_rate=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )


def load_image(path: Path) -> torch.Tensor:
    with PillowImage.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


def target(image: Image, flipped: bool) -> dict[str, torch.Tensor]:
    masks = np.zeros((len(image.masks), image.size), dtype=np.uint8)
    boxes: list[list[float]] = []
    for index, rle in enumerate(image.masks):
        encoded = pixels(rle, image.size)
        y, x = np.divmod(encoded, image.width)
        if flipped:
            x = image.width - 1 - x
        masks[index, y * image.width + x] = 1
        boxes.append([float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "masks": torch.from_numpy(masks.reshape(-1, image.height, image.width)),
    }


def build(config: Settings, pretrained: bool) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=config["min_size"],
        max_size=config["max_size"],
        box_detections_per_img=config["detections"],
        box_score_thresh=config["score_threshold"],
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if not pretrained:
        # COCO 版の ResNet は FrozenBatchNorm2d (eps=0) を使用する。
        for name, module in list(model.backbone.named_modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                parent, _, child = name.rpartition(".")
                model.backbone.get_submodule(parent).add_module(
                    child, FrozenBatchNorm2d(module.num_features, eps=0.0)
                )
    return model


def fit(
    rows: list[Image], config: Settings, device: torch.device, *, smoke: bool = False
) -> MaskRCNN:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build(config, pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
    batch_size = config["batch_size"]
    for epoch in range(1 if smoke else config["epochs"]):
        model.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            inputs: list[torch.Tensor] = []
            targets: list[dict[str, torch.Tensor]] = []
            for index in order[start : start + batch_size]:
                image = rows[index]
                flipped = bool(rng.random() < config["flip"])
                tensor = load_image(image.path)
                inputs.append((tensor.flip((2,)) if flipped else tensor).to(device))
                targets.append(
                    {name: value.to(device) for name, value in target(image, flipped).items()}
                )
            optimizer.zero_grad(set_to_none=True)
            losses = model(inputs, targets)
            loss = torch.stack(list(losses.values())).sum()
            if not torch.isfinite(loss).item():
                raise ValueError("学習損失が有限値ではありません")
            loss.backward()
            optimizer.step()
            if smoke:
                break
        schedule.step()
        print(f"epoch {epoch + 1} 完了", flush=True)
    return model


@torch.inference_mode()
def rankings(
    model: MaskRCNN, path: Path, config: Settings, device: torch.device, *, weighted: bool
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    prediction = model.eval()([load_image(path).to(device)])[0]
    scores = prediction["scores"].cpu().tolist()
    masks = prediction["masks"]
    if not torch.isfinite(masks).all().item():
        raise ValueError(f"マスクの値が不正です: {path}")
    probabilities = masks[:, 0].cpu().numpy()
    result: list[tuple[float, str]] = []
    rescored: list[tuple[float, str]] = []
    for score, probability in zip(scores, probabilities, strict=True):
        selected = probability >= config["mask_threshold"]
        rle = encode(np.flatnonzero(selected))
        if rle:
            result.append((score, rle))
            if weighted:
                rescored.append((score * float(probability[selected].mean()), rle))
    return result, rescored


def predict(
    model: MaskRCNN, path: Path, config: Settings, device: torch.device
) -> list[tuple[float, str]]:
    weighted = config.get("rank_by_mask", False)
    return rankings(model, path, config, device, weighted=weighted)[int(weighted)]
