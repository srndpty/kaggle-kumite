"""予測の妥当性と、検証の独立性の検査。"""

from collections.abc import Callable, Sequence

import numpy as np


def validate_probabilities(
    pred: np.ndarray, n_rows: int, n_classes: int, atol: float = 1e-6
) -> None:
    """多クラス確率として保存・提出してよいかを確かめる。不正なら ValueError。"""
    if n_rows == 0:
        raise ValueError("予測が 0 行です")
    if pred.shape != (n_rows, n_classes):
        raise ValueError(f"形状が {pred.shape} です（期待値 {(n_rows, n_classes)}）")
    if not np.isfinite(pred).all():
        raise ValueError("NaN または Inf を含みます")
    if (pred < 0).any() or (pred > 1).any():
        raise ValueError(f"確率が [0, 1] の範囲外です（最小 {pred.min()}、最大 {pred.max()}）")
    max_dev = float(np.abs(pred.sum(axis=1) - 1).max())
    if max_dev > atol:
        raise ValueError(f"確率の和が 1 からずれています（最大 {max_dev:.2e}）")


def shift_labels(y: np.ndarray, rows: np.ndarray, n_classes: int, seed: int = 0) -> np.ndarray:
    """rows の行のラベルを、必ず別のクラスに置き換えたコピーを返す。"""
    shifted = y.copy()
    offsets = np.random.default_rng(seed).integers(1, n_classes, size=int(rows.sum()))
    shifted[rows] = (y[rows] + offsets) % n_classes
    return shifted


def check_label_independence(
    build: Callable[[np.ndarray], Sequence[np.ndarray]],
    y: np.ndarray,
    y_perturbed: np.ndarray,
    atol: float = 1e-9,
) -> None:
    """ラベルを y_perturbed に変えても build の出力が変わらないことを確かめる。

    build は「隠すべき行（外側の検証行など）のラベルを読んではいけない処理」。y と y_perturbed は
    隠すべき行でだけ異なるようにする（`shift_labels`）。出力が変われば、その行のラベルが漏れている。
    """
    changed = np.flatnonzero(y != y_perturbed)
    if len(changed) == 0:
        raise ValueError("y_perturbed が y と同じです")
    base, perturbed = build(y), build(y_perturbed)
    for i, (a, b) in enumerate(zip(base, perturbed, strict=True)):
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=atol):
            diff = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
            raise AssertionError(f"出力 {i} が隠すべきラベルに依存しています（最大差 {diff:.3e}）")
