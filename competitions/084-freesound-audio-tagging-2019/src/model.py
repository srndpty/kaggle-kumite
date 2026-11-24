"""クリップ単位の CNN 学習と可変長音源の推論。"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from dataset import RATE, WINDOW, read_wave, windows
from torch import nn

from kumite.audio import Features, SmallCNN


class Tagger(nn.Module):
    def __init__(self, tags: int) -> None:
        super().__init__()
        self.features = Features(rate=RATE, bands=64)
        self.classifier = SmallCNN(tags)

    def forward(self, wave: torch.Tensor, *, time_mask: bool = False) -> torch.Tensor:
        features = self.features(wave)
        if time_mask:
            width = 20
            start = torch.randint(
                features.shape[-1] - width + 1, (len(wave), 1), device=wave.device
            )
            columns = torch.arange(features.shape[-1], device=wave.device)
            mask = (columns >= start) & (columns < start + width)
            means = features.mean(dim=-1, keepdim=True)
            features = torch.where(mask[:, None, None, :], means, features)
        return self.classifier(features)


def fit(
    curated: list[npt.NDArray[np.float32]],
    curated_labels: npt.NDArray[np.bool_],
    noisy_paths: list[Path],
    noisy_labels: npt.NDArray[np.bool_],
    *,
    updates: int,
    seed: int,
    device: torch.device,
    deadline: float,
    time_mask: bool,
    staged_noisy: bool = False,
) -> Tagger:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model = Tagger(curated_labels.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    if staged_noisy and (not noisy_paths or updates < 4):
        raise ValueError("Staged training requires noisy clips and at least four updates")
    mixed_noisy_count = 8 if noisy_paths and not staged_noisy else 0
    warmup_updates = updates // 4 if staged_noisy else 0
    for step in range(updates):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training deadline reached")
        noisy_count = 32 if step < warmup_updates else mixed_noisy_count
        clean_indices = rng.integers(len(curated), size=32 - noisy_count)
        wave_batch = [curated[i] for i in clean_indices]
        labels = curated_labels[clean_indices]
        if noisy_count:
            noisy_indices = rng.integers(len(noisy_paths), size=noisy_count)
            wave_batch.extend(read_wave(noisy_paths[i]) for i in noisy_indices)
            labels = np.concatenate((labels, noisy_labels[noisy_indices]))
        batch = np.zeros((len(wave_batch), WINDOW), dtype=np.float32)
        for i, wave in enumerate(wave_batch):
            start = int(rng.integers(max(1, len(wave) - WINDOW + 1)))
            crop = wave[start : start + WINDOW]
            batch[i, : len(crop)] = crop
        optimizer.zero_grad()
        logits = model(torch.from_numpy(batch).to(device), time_mask=time_mask)
        loss = criterion(logits, torch.as_tensor(labels, dtype=torch.float32, device=device))
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(
    model: Tagger,
    wave: npt.NDArray[np.float32],
    device: torch.device,
    *,
    maximum: bool,
) -> npt.NDArray[np.float64]:
    frames = windows(wave)
    batches = [
        model(torch.from_numpy(frames[i : i + 32]).to(device)).sigmoid()
        for i in range(0, len(frames), 32)
    ]
    scores = torch.cat(batches)
    aggregate = scores.amax(dim=0) if maximum else scores.mean(dim=0)
    return np.asarray(aggregate.cpu().numpy(), dtype=np.float64)
