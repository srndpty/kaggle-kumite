"""近重複の連結、学習側のクラス重み、提出用の確率を確認する。"""

from pathlib import Path

import numpy as np
import torch
from dataset import near_duplicate_groups
from model import class_weights, ensemble, network
from PIL import Image

from kumite.checks import validate_probabilities


def test_near_duplicates_are_connected_transitively() -> None:
    a, b, c, different = 0, 0b1111, 0b11111111, (1 << 64) - 1
    assert near_duplicate_groups([a, c, different, b, a]) == [0, 0, 1, 0, 0]


def test_weights_use_only_training_labels() -> None:
    labels = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4])
    train = np.arange(8)
    before = class_weights(labels[train])
    labels[8:] = 0
    torch.testing.assert_close(before, class_weights(labels[train]))
    torch.testing.assert_close(before, torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0]))


def test_checkpoint_replays_multiclass_probabilities(tmp_path: Path) -> None:
    image = tmp_path / "leaf.png"
    Image.new("RGB", (32, 32), (200, 100, 50)).save(image)
    checkpoint = tmp_path / "fold-0.pt"
    torch.save(network().state_dict(), checkpoint)
    probability = ensemble([checkpoint], [image], 32, torch.device("cpu"))
    validate_probabilities(probability, 1, 5)
