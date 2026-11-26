"""波形の前処理と 1D CNN。"""

from collections.abc import Iterable

import numpy as np
import torch
from dataset import SAMPLE_RATE, SHAPE
from torch import nn

SCALE = 1e20
BAND = (25, 500)
BAND_MASK = (np.fft.rfftfreq(SHAPE[1], 1 / SAMPLE_RATE) >= BAND[0]) & (
    np.fft.rfftfreq(SHAPE[1], 1 / SAMPLE_RATE) <= BAND[1]
)


def fit_whitening(batches: Iterable[np.ndarray]) -> np.ndarray:
    power = np.zeros((SHAPE[0], SHAPE[1] // 2 + 1))
    count = 0
    for batch in batches:
        power += np.square(np.abs(np.fft.rfft(batch * SCALE))).sum(axis=0)
        count += len(batch)
    if not count:
        raise ValueError("PSD の学習行がありません")
    power /= count
    floor = np.maximum(np.median(power[:, BAND_MASK], axis=1, keepdims=True) * 1e-6, 1e-12)
    return np.sqrt(np.maximum(power, floor) / SHAPE[1])


def transform(
    batch: np.ndarray,
    channels: int,
    bandpass: bool,
    whitening: np.ndarray | None,
) -> np.ndarray:
    values = batch[:, :channels] * SCALE
    if bandpass:
        spectrum = np.fft.rfft(values)
        spectrum[..., ~BAND_MASK] = 0
        if whitening is not None:
            spectrum /= whitening[:channels]
        values = np.fft.irfft(spectrum, n=SHAPE[1])
    return values.astype(np.float32)


def build_model(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(channels, 24, kernel_size=31, stride=8, padding=15),
        nn.BatchNorm1d(24),
        nn.GELU(),
        nn.Conv1d(24, 48, kernel_size=15, stride=4, padding=7),
        nn.BatchNorm1d(48),
        nn.GELU(),
        nn.Conv1d(48, 64, kernel_size=9, stride=2, padding=4),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    )


def tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(batch).to(device)
