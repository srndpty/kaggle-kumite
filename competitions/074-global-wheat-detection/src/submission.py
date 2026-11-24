"""検出結果を原画像座標の提出 CSV に書く。"""

import csv
import hashlib
from pathlib import Path

import torch
from model import build_model, detect
from PIL import Image


def submission_images(sample: Path, *, hidden_template_sha256: str | None = None) -> list[Path]:
    with sample.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "PredictionString"]:
            raise ValueError("Unexpected submission columns")
        ids = [row["image_id"] for row in reader]
    images = {path.stem: path for path in (sample.parent / "test").glob("*.jpg")}
    if not ids or not all(ids) or len(set(ids)) != len(ids) or not images:
        raise ValueError("Missing or duplicate submission IDs")
    if set(ids) == images.keys():
        return [images[key] for key in ids]
    # Kaggle の採点時は公開サンプルを保持したまま test 画像が差し替わる。
    if (
        hidden_template_sha256 is not None
        and hashlib.sha256(sample.read_bytes()).hexdigest() == hidden_template_sha256
        and len(images) > len(ids)
        and images.keys().isdisjoint(ids)
    ):
        return [images[key] for key in sorted(images)]
    raise ValueError("Test JPEG IDs do not match submission template")


def write_submission(
    sample: Path,
    checkpoint: Path,
    output: Path,
    size: int,
    nms: float,
    device: torch.device,
    *,
    flip_fusion: bool = False,
    hidden_template_sha256: str | None = None,
) -> None:
    images = submission_images(sample, hidden_template_sha256=hidden_template_sha256)
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
