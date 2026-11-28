"""マスク付き二値交差エントロピーで小型 CNN を学習する。"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dataset import RATE, SPECIES, WINDOW, Example, aggregate, crop, read_wave, starts
from torch import nn

from kumite.audio import Features, SmallCNN

BATCH = 16


def model() -> nn.Sequential:
    return nn.Sequential(Features(rate=RATE, bands=64), SmallCNN(len(SPECIES)))


def read_crop(path: Path, start: int) -> np.ndarray:
    with sf.SoundFile(path) as recording:
        if recording.samplerate != RATE or recording.channels != 1:
            raise ValueError(f"学習用 PCM は 32 kHz モノラルが必要です: {path}")
        recording.seek(start)
        samples = recording.read(WINDOW, dtype="float32")
    return crop(samples, 0)


def fit(
    paths: list[Path],
    pools: tuple[list[Example], list[Example], list[Example]],
    updates: int,
    seed: int,
    device: torch.device,
) -> nn.Sequential:
    rng = np.random.default_rng(seed)
    negative_rng = np.random.default_rng(seed + 1)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    network = model().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    positive, background, hard = pools
    for _ in range(updates):
        # 正例数と負例数を全比較で固定する。
        n_hard = 4 if hard else 0
        selected = [
            *(positive[i] for i in rng.integers(len(positive), size=8)),
            *(background[i] for i in negative_rng.integers(len(background), size=8 - n_hard)),
        ]
        if hard:
            selected.extend(hard[i] for i in negative_rng.integers(len(hard), size=n_hard))
        wave = np.stack([read_crop(paths[item.row], item.start) for item in selected])
        labels = np.zeros((BATCH, len(SPECIES)), dtype=np.float32)
        weights = np.zeros_like(labels)
        for row, item in enumerate(selected):
            if row < 8:
                labels[row, list(item.species)] = 1
                weights[row, list(item.species)] = 1
            elif item.species:
                weights[row, list(item.species)] = 0.25
            else:
                weights[row] = 0.25 / len(SPECIES)
        logits = network(torch.from_numpy(wave).to(device))
        expected = torch.from_numpy(labels).to(device)
        mask = torch.from_numpy(weights).to(device)
        loss = (
            nn.functional.binary_cross_entropy_with_logits(logits, expected, reduction="none")
            * mask
        ).sum() / BATCH
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return network.eval()


@torch.inference_mode()
def predict(
    network: nn.Sequential, path: Path, hop: int, top: int, device: torch.device
) -> np.ndarray:
    wave = read_wave(path)
    grid = starts(len(wave), hop)
    values = []
    for offset in range(0, len(grid), BATCH):
        windows = np.stack([crop(wave, start) for start in grid[offset : offset + BATCH]])
        values.append(network(torch.from_numpy(windows).to(device)).sigmoid().cpu().numpy())
    return aggregate(np.concatenate(values).astype(np.float64), top)
