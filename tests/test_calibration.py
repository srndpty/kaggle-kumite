import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from kumite.calibration import apply_temperature, fit_temperature


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _logloss(y: np.ndarray, proba: np.ndarray) -> float:
    return float(-np.log(proba[np.arange(len(y)), y]).mean())


def test_apply_temperature_one_is_identity_and_rows_sum_to_one() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(apply_temperature(proba, 1.0), proba)
    np.testing.assert_allclose(apply_temperature(proba, 2.5).sum(axis=1), 1.0)


def test_apply_temperature_above_one_softens() -> None:
    proba = np.array([[0.9, 0.05, 0.05]])
    assert apply_temperature(proba, 2.0)[0, 0] < 0.9
    assert apply_temperature(proba, 0.5)[0, 0] > 0.9


def test_fit_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20000, 4)) * 2
    y = np.array([rng.choice(4, p=p) for p in _softmax(logits)])
    overconfident = _softmax(logits * 2)  # 真の確率を T=0.5 で鋭くした予測
    assert fit_temperature(y, overconfident, _logloss) == pytest.approx(2.0, rel=0.05)


def test_fit_temperature_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="行数"):
        fit_temperature(np.zeros(3, dtype=int), np.full((2, 2), 0.5), _logloss)


def _scipy_temperature(y: np.ndarray, proba: np.ndarray, bounds: tuple[float, float]) -> float:
    """旧実装（#001 の calibrate.py）: T 上の scipy の有界 Brent 法。回帰比較の基準。"""
    optimize = importlib.import_module("scipy.optimize")
    minimize: Callable[..., Any] = optimize.minimize_scalar

    def loss(t: float) -> float:
        return _logloss(y, apply_temperature(proba, float(t)))

    result = minimize(loss, bounds=bounds, method="bounded")
    x: float = result.x
    return float(x)


def _case(scale: float, n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """真の確率から y を引き、logit を scale 倍した予測（scale > 1 で過信、< 1 で控えめ）。"""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 5)) * 2
    y = np.array([rng.choice(5, p=p) for p in _softmax(logits)])
    return y, _softmax(logits * scale)


@pytest.mark.parametrize(
    ("scale", "at_bound"),
    [
        (2.0, None),  # 典型的な過信（T ≈ 2）
        (1.0, None),  # 較正済み（T ≈ 1）
        (12.0, 5.0),  # 最適な T が上限 5.0 を超える → 上限に張り付く
        (0.01, 0.2),  # 最適な T が下限 0.2 を下回る → 下限に張り付く
    ],
)
def test_fit_temperature_matches_scipy_bounded(scale: float, at_bound: float | None) -> None:
    y, proba = _case(scale)
    new = fit_temperature(y, proba, _logloss)
    old = _scipy_temperature(y, proba, (0.2, 5.0))
    assert new == pytest.approx(old, rel=1e-3)
    loss_new = _logloss(y, apply_temperature(proba, new))
    loss_old = _logloss(y, apply_temperature(proba, old))
    # 内部の最適値では一致し、境界では新しい実装の方が境界に近い（損失は同じか小さい）
    assert loss_new <= loss_old + 1e-7
    assert loss_new == pytest.approx(loss_old, abs=1e-5)
    if at_bound is not None:
        assert new == pytest.approx(at_bound, rel=1e-4)


def test_fit_temperature_handles_zero_and_tiny_probabilities() -> None:
    y, proba = _case(3.0)
    proba = proba.copy()
    proba[:100] = 0.0
    proba[np.arange(100), (y[:100] + 1) % 5] = 1.0  # 自信のある誤り（正解の確率 0）
    proba[100:200, :] = np.array([1 - 4e-300, 1e-300, 1e-300, 1e-300, 1e-300])

    def clipped(y_: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return _logloss(y_, p / p.sum(axis=1, keepdims=True))

    new = fit_temperature(y, proba, clipped)
    assert np.isfinite(new)
    assert 0.2 <= new <= 5.0
    assert np.isfinite(apply_temperature(proba, new)).all()
    assert clipped(y, apply_temperature(proba, new)) <= clipped(y, proba)
