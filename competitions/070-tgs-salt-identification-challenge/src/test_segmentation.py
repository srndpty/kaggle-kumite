"""採点・提出・ラベル独立性の回帰テスト。"""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import train
from dataset import Data
from models import fit, load_checkpoint, predict
from segmentation import SIZE, duplicate_groups, encode, image_scores, make_folds, write_submission

from kumite.checks import check_label_independence


def test_metric_empty_and_strict_boundary() -> None:
    target = np.array([[[0, 0]], [[1, 0]], [[0, 0]], [[1, 0]], [[1, 0]]], dtype=bool)
    prediction = np.array([[[0, 0]], [[0, 0]], [[1, 0]], [[1, 1]], [[1, 0]]], dtype=bool)
    np.testing.assert_array_equal(image_scores(target, prediction), [1, 0, 0, 0, 1])


def test_rle_and_id_order(tmp_path: Path) -> None:
    masks = np.zeros((2, SIZE, SIZE), dtype=bool)
    masks[0, 1:3, 0] = True
    masks[0, 0, 2] = True
    encoded = encode(masks[0])
    assert encoded == "2 2 203 1"
    decoded = np.zeros(SIZE**2, dtype=bool)
    numbers = list(map(int, encoded.split()))
    for start, length in zip(numbers[::2], numbers[1::2], strict=True):
        decoded[start - 1 : start - 1 + length] = True
    np.testing.assert_array_equal(decoded.reshape((SIZE, SIZE), order="F"), masks[0])
    path = tmp_path / "submission.csv"
    write_submission(path, ["z", "a"], masks)
    with path.open(newline="") as file:
        assert list(csv.reader(file)) == [["id", "rle_mask"], ["z", encoded], ["a", ""]]


def test_groups_and_threshold_independence(monkeypatch: pytest.MonkeyPatch) -> None:
    images = np.random.default_rng(42).integers(0, 255, (80, 6, 6), dtype=np.uint8)
    images[1] = images[0, :, ::-1]
    masks = np.zeros_like(images, dtype=bool)
    for i, rows in enumerate((0, 1, 2, 4)):
        masks[20 * i : 20 * (i + 1), :rows, : (2 if rows == 1 else 6)] = True
    groups = duplicate_groups(images)
    folds = make_folds(masks, groups, 5)
    assert folds[0] == folds[1]
    assert all(len(set(folds[groups == group])) == 1 for group in set(groups))
    outer_train = np.flatnonzero(folds != 0)
    data = Data([], [], images, masks, images[:0], np.zeros(len(images)))
    monkeypatch.setattr(train, "fit", lambda *_: None)
    monkeypatch.setattr(train, "predict", lambda _, pixels, __: pixels / 255)
    changed = masks.copy()
    changed[folds == 0] ^= True
    check_label_independence(
        lambda labels: [
            np.array(
                train.calibrate(
                    replace(data, masks=labels),
                    outer_train,
                    groups,
                    torch.device("cpu"),
                )
            )
        ],
        masks,
        changed,
    )


def test_cpu_epoch(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    pixels = np.zeros((2, SIZE, SIZE), dtype=np.uint8)
    device = torch.device("cpu")
    model = fit(pixels, pixels.astype(bool), device, "baseline", epochs=1)
    original = predict(model, pixels, device)
    assert original.shape == pixels.shape
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = load_checkpoint(checkpoint, device)
    np.testing.assert_array_equal(predict(restored, pixels, device), original)
