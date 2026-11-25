"""ResNet-18 を encoder に使う4クラス U-Net。"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet


def normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = images.to(device=device, dtype=torch.float32) / 255
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def make_encoder(pretrained: Path | None) -> ResNet:
    encoder = resnet18()
    if pretrained is not None:
        encoder.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
    return encoder


def make_presence_model(pretrained: Path | None = None) -> ResNet:
    encoder = make_encoder(pretrained)
    encoder.fc = nn.Linear(encoder.fc.in_features, 4)
    return encoder


class SteelUNet(nn.Module):
    def __init__(self, pretrained: Path | None = None) -> None:
        super().__init__()
        encoder = make_encoder(pretrained)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.first = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.second = encoder.layer2
        self.third = encoder.layer3
        self.bottom = encoder.layer4
        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(current + skip, skip, 3, padding=1, bias=False),
                nn.BatchNorm2d(skip),
                nn.ReLU(inplace=True),
            )
            for current, skip in ((512, 256), (256, 128), (128, 64), (64, 64))
        )
        self.head = nn.Conv2d(64, 4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        stem = self.stem(images)
        first = self.first(stem)
        second = self.second(first)
        third = self.third(second)
        features = self.bottom(third)
        for block, skip in zip(self.decoder, (third, second, first, stem), strict=True):
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat((features, skip), dim=1))
        return F.interpolate(
            self.head(features), size=images.shape[-2:], mode="bilinear", align_corners=False
        )


def training_loss(logits: torch.Tensor, target: torch.Tensor, *, use_dice: bool) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if not use_dice:
        return bce
    probability = logits.sigmoid()
    positive = target.sum(dim=(-2, -1)) > 0
    intersection = (probability * target).sum(dim=(-2, -1))
    mass = (probability + target).sum(dim=(-2, -1))
    soft_dice = ((1 - (2 * intersection + 1) / (mass + 1)) * positive).sum()
    combined = (bce + soft_dice / positive.sum().clamp_min(1)) / 2
    return torch.where(positive.any(), combined, bce)
