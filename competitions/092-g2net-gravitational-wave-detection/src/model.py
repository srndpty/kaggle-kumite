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
WHITENING_SHAPE = (SHAPE[0], SHAPE[1] // 2 + 1)


def validate_batch(batch: np.ndarray) -> None:
    if batch.ndim != 3 or batch.shape[1:] != SHAPE or batch.dtype.kind != "f":
        raise ValueError(f"波形バッチの形状または型が不正です: {batch.shape}, {batch.dtype}")
    if not np.isfinite(batch).all():
        raise ValueError("波形バッチに NaN または Inf があります")


def fit_whitening(batches: Iterable[np.ndarray]) -> np.ndarray:
    power = np.zeros(WHITENING_SHAPE)
    count = 0
    for batch in batches:
        validate_batch(batch)
        try:
            with np.errstate(over="raise", invalid="raise"):
                power += np.square(np.abs(np.fft.rfft(batch * SCALE))).sum(axis=0)
        except FloatingPointError as exc:
            raise ValueError("波形の振幅が大きすぎて PSD を計算できません") from exc
        count += len(batch)
    if not count:
        raise ValueError("PSD の学習行がありません")
    power /= count
    if not np.isfinite(power).all():
        raise ValueError("PSD に NaN または Inf があります")
    floor = np.maximum(np.median(power[:, BAND_MASK], axis=1, keepdims=True) * 1e-6, 1e-12)
    return np.sqrt(np.maximum(power, floor) / SHAPE[1])


def transform(
    batch: np.ndarray,
    channels: int,
    bandpass: bool,
    whitening: np.ndarray | None,
) -> np.ndarray:
    validate_batch(batch)
    if not 1 <= channels <= SHAPE[0]:
        raise ValueError("使用するチャネル数が不正です")
    if whitening is not None and (
        not bandpass
        or whitening.shape != WHITENING_SHAPE
        or not np.isfinite(whitening).all()
        or (whitening <= 0).any()
    ):
        raise ValueError("白色化係数または帯域制限の設定が不正です")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values = batch[:, :channels] * SCALE
            if bandpass:
                spectrum = np.fft.rfft(values)
                spectrum[..., ~BAND_MASK] = 0
                if whitening is not None:
                    spectrum /= whitening[:channels]
                values = np.fft.irfft(spectrum, n=SHAPE[1])
            values = values.astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError("有限な float32 波形に変換できません") from exc
    if not np.isfinite(values).all():
        raise ValueError("変換後の波形に NaN または Inf があります")
    return values


class GlobalMaxPool(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.amax(dim=-1)


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
        GlobalMaxPool(),
        nn.Linear(64, 1),
    )


def tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(batch).to(device)
