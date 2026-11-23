"""学習と提出で共有する画像モデルと推論。"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

N_CLASSES = 5
BATCH_SIZE = 32


class Images(Dataset[tuple[Tensor, int]]):
    def __init__(
        self, paths: Sequence[Path], labels: Sequence[int], size: int, flip: bool = False
    ) -> None:
        self.paths = paths
        self.labels = labels
        steps: list[object] = [transforms.Resize((size, size))]
        if flip:
            steps.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(
            [
                *steps,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image.convert("RGB"))
        if not isinstance(pixels, Tensor):
            raise TypeError("画像の前処理がテンソルを返しませんでした")
        return pixels, self.labels[index]


def network(pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def class_weights(labels: np.ndarray) -> Tensor:
    counts = np.bincount(labels, minlength=N_CLASSES)
    if (counts == 0).any():
        raise ValueError("学習 fold に存在しないクラスがあります")
    return torch.as_tensor(counts, dtype=torch.float32).rsqrt()


@torch.inference_mode()
def predict(
    model: nn.Module, paths: Sequence[Path], size: int, device: torch.device, flip: bool = False
) -> np.ndarray:
    model.eval()
    loader = DataLoader(Images(paths, [0] * len(paths), size), batch_size=BATCH_SIZE)
    batches = []
    for images, _ in loader:
        images = images.to(device)
        probabilities = model(images).softmax(1)
        if flip:
            probabilities = (probabilities + model(images.flip((-1,))).softmax(1)) / 2
        batches.append(probabilities.cpu().numpy())
    return np.concatenate(batches)


def ensemble(
    checkpoints: Sequence[Path],
    paths: Sequence[Path],
    size: int,
    device: torch.device,
    flip: bool = False,
) -> np.ndarray:
    model = network().to(device)
    predictions = []
    for path in checkpoints:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        predictions.append(predict(model, paths, size, device, flip))
    return np.mean(predictions, axis=0)
