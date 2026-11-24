"""音声の対数スペクトル特徴量と小型 CNN。"""

import librosa
import numpy as np
import torch
from torch import nn


class Features(nn.Module):
    weights: torch.Tensor
    window: torch.Tensor

    def __init__(self, linear: bool = False, *, rate: int = 16000, bands: int = 40):
        super().__init__()
        if linear:
            frequencies = np.linspace(0, rate / 2, 257)
            edges = np.linspace(0, rate / 2, bands + 2)
            weights = np.maximum(
                0,
                np.minimum(
                    (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None],
                    (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None],
                ),
            )
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        else:
            weights = librosa.filters.mel(sr=rate, n_fft=512, n_mels=bands)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(400))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            wave,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        return torch.log(torch.matmul(self.weights, power) + 1e-6).unsqueeze(1)


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 12):
        super().__init__()
        layers: list[nn.Module] = []
        channels = 1
        for output in (32, 64, 128):
            layers.extend(
                (
                    nn.Conv2d(channels, output, 3, padding=1),
                    nn.BatchNorm2d(output),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
            )
            channels = output
        layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
