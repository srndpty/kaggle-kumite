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
