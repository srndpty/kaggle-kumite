"""GPU augmentation・入力の読み込み・指標の単体テスト（CPU でも動く小さな固定データ）。

実行: uv run pytest competitions/060-state-farm-distracted-driver-detection/src
"""

import numpy as np
import pytest
import torch
from models import DEVICE, ModelConfig, augment, image_batches, row_logloss


def _image(h: int = 24, w: int = 32) -> torch.Tensor:
    """左半分が明るく右半分が暗い (1, 3, h, w) の画像。"""
    x = torch.zeros(1, 3, h, w, device=DEVICE)
    x[..., : w // 2] = 0.8
    return x


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def test_no_augmentation_is_identity() -> None:
    x = _image()
    assert torch.equal(augment(x, ModelConfig(aug="none"), _gen()), x)


def test_hflip_swaps_left_and_right() -> None:
    x = _image().repeat(64, 1, 1, 1)
    out = augment(x, ModelConfig(aug="none", hflip=True), _gen())
    left = out[..., :16].mean(dim=(1, 2, 3))
    flipped = left < 0.4  # 反転した画像は左半分が暗い
    assert 0 < int(flipped.sum()) < 64  # 確率 0.5 なので、64 枚中で両方が出る
    torch.testing.assert_close(out[flipped], x[flipped].flip(-1))
    torch.testing.assert_close(out[~flipped], x[~flipped])


def test_weak_augmentation_keeps_shape_range_and_rough_layout() -> None:
    x = _image().repeat(32, 1, 1, 1)
    out = augment(x, ModelConfig(aug="weak"), _gen())
    assert out.shape == x.shape
    assert float(out.min()) >= 0 and float(out.max()) <= 1
    # 回転 ±10° と切り出しでは左右の明暗は入れ替わらない（反転しない）
    assert bool((out[..., :8].mean(dim=(1, 2, 3)) > out[..., -8:].mean(dim=(1, 2, 3))).all())
    assert not torch.equal(out[0], out[1])  # 画像ごとに別の乱数


def test_same_seed_gives_same_augmentation() -> None:
    x = _image().repeat(4, 1, 1, 1)
    cfg = ModelConfig(aug="weak", hflip=True)
    torch.testing.assert_close(augment(x, cfg, _gen(1)), augment(x, cfg, _gen(1)))


def test_batches_keep_requested_row_order() -> None:
    images = np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1).repeat(3, axis=3)
    rows = np.array([3, 0, 4, 1])
    got = torch.cat(list(image_batches(images, rows, batch_size=3)))
    np.testing.assert_allclose(got[:, 0, 0, 0].cpu().numpy() * 255, rows, atol=1e-4)


def test_row_logloss_clips_and_renormalizes() -> None:
    proba = np.array([[1.0, 0.0], [0.5, 0.5]])
    loss = row_logloss(np.array([1, 0]), proba)
    assert loss[0] == pytest.approx(-np.log(1e-15), rel=1e-6)
    assert loss[1] == pytest.approx(np.log(2))
