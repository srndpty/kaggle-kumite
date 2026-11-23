"""検出結果を原画像座標の提出 CSV に書く。"""

import csv
from pathlib import Path

import torch
from model import build_model, detect
from PIL import Image


def write_submission(
    test: Path,
    checkpoint: Path,
    output: Path,
    size: int,
    nms: float,
    device: torch.device,
    *,
    flip_fusion: bool = False,
) -> None:
    images = sorted(test.glob("*.jpg"))
    if not images:
        raise ValueError("No test JPEGs found")
    model = build_model(size, nms, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "PredictionString"))
        for path in images:
            with Image.open(path) as image:
                width, height = image.size
            boxes, scores = detect(model, path, device, flip_fusion=flip_fusion)
            detections: list[str] = []
            for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
                left, top = int(x1), int(y1)
                right, bottom = min(width, int(x2 + 0.5)), min(height, int(y2 + 0.5))
                if right > left and bottom > top:
                    detections.append(f"{score:.8f} {left} {top} {right - left} {bottom - top}")
            writer.writerow((path.stem, " ".join(detections)))
