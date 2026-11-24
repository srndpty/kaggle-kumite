"""Faster R-CNN の学習・推論。"""

import random
from pathlib import Path

import numpy as np
import torch
from dataset import Sample
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d
from torchvision.transforms import functional as F


def build_model(size: int, nms: float, *, pretrained: bool) -> FasterRCNN:
    model = FasterRCNN(
        resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=3),
        num_classes=2,
        min_size=size,
        max_size=size,
        box_score_thresh=0.25,
        box_nms_thresh=nms,
        box_detections_per_img=300,
    )
    if pretrained:
        initial = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        initial.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
        model.load_state_dict(initial.state_dict())
    for layer in model.modules():
        if isinstance(layer, FrozenBatchNorm2d):
            layer.eps = 0.0
    return model


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return F.to_tensor(image.convert("RGB"))


def fit(
    samples: list[Sample],
    size: int,
    color: bool,
    device: torch.device,
    *,
    epochs: int = 8,
    pretrained: bool = True,
) -> FasterRCNN:
    torch.manual_seed(42)
    rng = random.Random(42)
    color_rng = random.Random(43)
    model = build_model(size, 0.5, pretrained=pretrained).to(device)
    optimizer = torch.optim.SGD(
        (param for param in model.parameters() if param.requires_grad),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    indices = list(range(len(samples)))
    model.train()
    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), 2):
            images: list[torch.Tensor] = []
            targets: list[dict[str, torch.Tensor]] = []
            for index in indices[start : start + 2]:
                sample = samples[index]
                image = load_image(sample.path)
                boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
                if rng.random() < 0.5:
                    image = F.hflip(image)
                    boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                if color:
                    image = F.adjust_brightness(image, color_rng.uniform(0.8, 1.2))
                    image = F.adjust_contrast(image, color_rng.uniform(0.8, 1.2))
                images.append(image.to(device))
                targets.append(
                    {
                        "boxes": boxes.to(device),
                        "labels": torch.ones(len(boxes), dtype=torch.long, device=device),
                    }
                )
            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = model(images, targets)
            loss = torch.stack(tuple(losses.values())).sum()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at batch {start}")
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{epochs} complete", flush=True)
    return model


@torch.inference_mode()
def detect(
    model: FasterRCNN, path: Path, device: torch.device, *, flip_fusion: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    image = load_image(path)
    model.eval()
    inputs = [image.to(device)]
    if flip_fusion:
        inputs.append(F.hflip(inputs[0]))
    result: list[dict[str, torch.Tensor]] = model(inputs)
    boxes = result[0]["boxes"].cpu().numpy().astype(np.float64)
    scores = result[0]["scores"].cpu().numpy().astype(np.float64)
    height, width = image.shape[-2:]
    if flip_fusion:
        flipped = result[1]["boxes"].cpu().numpy().astype(np.float64)
        flipped[:, [0, 2]] = width - flipped[:, [2, 0]]
        boxes, scores = fuse_views(
            np.concatenate((boxes, flipped)),
            np.concatenate((scores, result[1]["scores"].cpu().numpy().astype(np.float64))),
        )
    if (
        boxes.shape != (len(scores), 4)
        or scores.ndim != 1
        or not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or (boxes < 0).any()
        or (boxes[:, [0, 2]] > width).any()
        or (boxes[:, [1, 3]] > height).any()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError(f"Invalid detections: {path}")
    return boxes, scores


def fuse_views(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU 0.55 を超える矩形を信頼度で加重平均する。"""
    if len(boxes) == 0:
        return boxes, scores
    fused = np.empty_like(boxes)
    totals = np.zeros(len(boxes))
    counts = np.zeros(len(boxes), dtype=np.int64)
    used = 0
    for index in np.argsort(-scores, kind="stable"):
        box, score = boxes[index], scores[index]
        current = fused[:used]
        intersection = np.maximum(
            np.minimum(box[2:], current[:, 2:]) - np.maximum(box[:2], current[:, :2]), 0
        ).prod(axis=1)
        union = (box[2:] - box[:2]).prod() + (current[:, 2:] - current[:, :2]).prod(axis=1)
        overlap = intersection / (union - intersection)
        match = int(overlap.argmax()) if used else 0
        if used and overlap[match] > 0.55:
            fused[match] = (fused[match] * totals[match] + box * score) / (totals[match] + score)
        else:
            match = used
            fused[match] = box
            used += 1
        totals[match] += score
        counts[match] += 1
    confidence = totals[:used] / np.maximum(counts[:used], 2)
    order = np.argsort(-confidence, kind="stable")
    keep = order[confidence[order] >= 0.25][:300]
    return fused[keep], confidence[keep]
