"""固定epochの小型CNN。検証ラベルは学習関数へ渡さない。"""

import random
import time
from importlib.metadata import version
from typing import Any

import librosa
import numpy as np
import torch
from dataset import RATE
from supervision import collapse_probabilities
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Features(nn.Module):
    weights: torch.Tensor
    window: torch.Tensor

    def __init__(self, linear: bool = False):
        super().__init__()
        if linear:
            frequencies = np.linspace(0, RATE / 2, 257)
            edges = np.linspace(0, RATE / 2, 42)
            weights = np.maximum(
                0,
                np.minimum(
                    (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None],
                    (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None],
                ),
            )
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        else:
            weights = librosa.filters.mel(sr=RATE, n_fft=512, n_mels=40)
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


def shift_wave(wave: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(wave.shape[1], device=wave.device)[None, :] - shifts[:, None]
    valid = (positions >= 0) & (positions < wave.shape[1])
    return wave.gather(1, positions.clamp(0, wave.shape[1] - 1)) * valid


def mix_noise(wave: torch.Tensor, noise: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    selected = noise[
        torch.randint(len(noise), (len(wave),), generator=generator, device=wave.device)
    ]
    signal_rms = wave.square().mean(1, keepdim=True).sqrt()
    noise_rms = selected.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
    snr = 5 + 15 * torch.rand((len(wave), 1), generator=generator, device=wave.device)
    enabled = torch.rand((len(wave), 1), generator=generator, device=wave.device) < 0.5
    return (wave + selected * signal_rms / noise_rms * 10 ** (-snr / 20) * enabled).clamp(-1, 1)


@torch.no_grad()
def feature_batches(
    waves: np.ndarray | torch.Tensor,
    extractor: Features,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    pieces = []
    for start in range(0, len(waves), batch_size):
        values = waves[start : start + batch_size]
        batch = (
            values.to(device)
            if isinstance(values, torch.Tensor)
            else torch.tensor(np.array(values), device=device)
        )
        pieces.append(extractor(batch))
    return torch.cat(pieces)


def fit_model(
    features: torch.Tensor,
    waves: torch.Tensor,
    labels: np.ndarray,
    train_idx: np.ndarray,
    extractor: Features,
    experiment: str,
    seed: int,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    deadline: float | None = None,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, list[float]]:
    """全ラベル配列から学習行だけを取り出す。外側検証ラベルは参照しない。"""
    seed_all(seed)
    device = features.device
    index = torch.as_tensor(train_idx, device=device)
    y = torch.as_tensor(labels[train_idx], dtype=torch.long, device=device)
    coarse_y = y.clamp_max(11)
    counts = torch.bincount(coarse_y, minlength=12)
    if bool((counts == 0).any()):
        raise ValueError("学習側に欠けたクラスがあります")
    weights = counts[coarse_y].float().reciprocal()
    # 学習foldだけで周波数別の標準化統計を計算する。
    mean = features[index].mean(dim=(0, 3), keepdim=True)
    std = features[index].std(dim=(0, 3), keepdim=True).clamp_min(1e-5)
    model = SmallCNN(31 if experiment == "E05" else 12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device=device).manual_seed(seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 10000)
    noise = waves[index[y == 10]]
    losses = []
    for epoch in range(epochs):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("事前に固定した総実行時間予算を超えました")
        model.train()
        order = torch.multinomial(weights, len(index), replacement=True, generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            rows = index[selected]
            if experiment in ("E03", "E04"):
                wave = waves[rows]
                if experiment == "E03":
                    shifts = torch.randint(
                        -1600, 1601, (len(rows),), device=device, generator=augmentation_generator
                    )
                    wave = shift_wave(wave, shifts)
                else:
                    wave = mix_noise(wave, noise, augmentation_generator)
                batch = extractor(wave)
            else:
                batch = features[rows]
            logits = model((batch - mean) / std)
            loss = nn.functional.cross_entropy(logits, y[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        losses.append(total / len(order))
        print(f"epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", flush=True)
    return model, mean, std, losses


@torch.no_grad()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(features), batch_size):
        logits = model((features[start : start + batch_size] - mean) / std)
        result.append(logits.softmax(dim=1).cpu().numpy())
    return collapse_probabilities(np.concatenate(result))


def environment() -> dict[str, Any]:
    return dict(
        torch=torch.__version__,
        numpy=np.__version__,
        librosa=version("librosa"),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
