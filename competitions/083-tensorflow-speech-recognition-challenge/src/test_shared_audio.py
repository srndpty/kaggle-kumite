"""共通化前の #083 の特徴量とチェックポイント契約を固定する。"""

import librosa
import numpy as np
import torch
from network import Features, SmallCNN
from torch import nn


def test_legacy_log_mel_features() -> None:
    wave = torch.linspace(-1, 1, 16000).reshape(1, -1)
    weights = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    power = (
        torch.stft(wave, 512, 160, 400, torch.hann_window(400), return_complex=True).abs().square()
    )
    expected = (torch.from_numpy(weights) @ power + 1e-6).log().unsqueeze(1)
    torch.testing.assert_close(Features()(wave), expected, atol=0, rtol=0)


def test_legacy_checkpoint_names_shapes_and_predictions() -> None:
    legacy = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 12),
    ).eval()
    checkpoint = {f"layers.{name}": value for name, value in legacy.state_dict().items()}
    current = SmallCNN().eval()
    assert {k: v.shape for k, v in checkpoint.items()} == {
        k: v.shape for k, v in current.state_dict().items()
    }
    current.load_state_dict(checkpoint, strict=True)
    features = torch.linspace(-1, 1, 40 * 101).reshape(1, 1, 40, 101)
    with torch.inference_mode():
        torch.testing.assert_close(current(features), legacy(features), atol=0, rtol=0)
