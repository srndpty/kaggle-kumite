"""小型 U-Net と fold 単位の学習・推論。"""

from pathlib import Path

import numpy as np
import torch
from segmentation import SEED, SIZE
from torch import nn
from torch.nn import functional as F

BATCH = 32
EPOCHS = 25


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (32, 64, 128, 256, 512)
        self.down = nn.ModuleList(
            block(a, b) for a, b in zip((1, *channels[:-1]), channels, strict=True)
        )
        self.up = nn.ModuleList(
            block(a + b, b) for a, b in zip(channels[:0:-1], channels[-2::-1], strict=True)
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.pad(images, (0, 112 - SIZE, 0, 112 - SIZE))
        skips: list[torch.Tensor] = []
        for layer in self.down:
            if skips:
                features = F.max_pool2d(features, 2)
            features = layer(features)
            skips.append(features)
        for layer, skip in zip(self.up, skips[-2::-1], strict=True):
            features = F.interpolate(features, scale_factor=2, mode="nearest")
            features = layer(torch.cat((features, skip), dim=1))
        return self.head(features)[:, 0, :SIZE, :SIZE]


def fit(
    images: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    variant: str,
    epochs: int = EPOCHS,
) -> UNet:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = UNet().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.as_tensor(images, device=device)
    y = torch.as_tensor(masks, device=device)
    for epoch in range(epochs):
        order = rng.permutation(len(images))
        for start in range(0, len(order), BATCH):
            ids = torch.as_tensor(order[start : start + BATCH], device=device)
            pixels = x[ids].float() / 255
            target = y[ids].float()
            if variant == "flip":
                flip = torch.rand((len(ids), 1, 1), device=device) < 0.5
                pixels = torch.where(flip, pixels.flip(-1), pixels)
                target = torch.where(flip, target.flip(-1), target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(pixels[:, None])
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if variant == "dice":
                prob = logits.sigmoid()
                intersection = (prob * target).sum(dim=(1, 2))
                mass = (prob + target).sum(dim=(1, 2))
                dice = (1 - (2 * intersection + 1) / (mass + 1)).mean()
                loss = (loss + dice) / 2
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{epochs}", flush=True)
    return model


@torch.inference_mode()
def predict(model: UNet, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = np.empty((len(images), SIZE, SIZE), dtype=np.float32)
    for start in range(0, len(images), BATCH):
        pixels = torch.as_tensor(images[start : start + BATCH], device=device).float() / 255
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(pixels[:, None])
        probabilities[start : start + len(pixels)] = logits.float().sigmoid().cpu().numpy()
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("画素確率が不正です")
    return probabilities


def load_checkpoint(path: Path, device: torch.device) -> UNet:
    model = UNet().to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def segment(
    model: UNet, images: np.ndarray, device: torch.device, threshold: float, variant: str
) -> np.ndarray:
    probabilities = predict(model, images, device)
    if variant == "flip-tta":
        reflected = predict(model, images[:, :, ::-1].copy(), device)
        probabilities = (probabilities + reflected[:, :, ::-1]) / 2
    return probabilities >= threshold
