"""M5 の Weighted Scaled Pinball Loss と区間指標を計算する。"""

import numpy as np
from model import QUANTILES, validate_prediction

from kumite.m5 import HORIZON, LEVELS, Hierarchy, Sales, historical_change_scale, revenue_weights


def scale_at(data: Sales, hierarchy: Hierarchy, origin: int) -> np.ndarray:
    return historical_change_scale(data, hierarchy, origin, squared=False)


def score(
    actual: np.ndarray,
    prediction: np.ndarray,
    hierarchy: Hierarchy,
    weight: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    validate_prediction(prediction, len(hierarchy.levels))
    if actual.shape != prediction.shape[:2]:
        raise ValueError("実績と予測の形状が一致しません")
    active = weight > 0
    if (scale[active] <= 0).any():
        raise ValueError("正の重みを持つ系列の WSPL の尺度を定義できません")
    scaled_weight = np.divide(weight, scale, out=np.zeros_like(weight), where=active)
    scores = []
    for index, quantile in enumerate(QUANTILES):
        error = actual - prediction[:, :, index]
        loss = np.maximum(quantile * error, (quantile - 1) * error).mean(axis=1)
        scores.append(
            np.bincount(hierarchy.levels, weights=loss * scaled_weight, minlength=len(LEVELS))
        )
    return np.stack(scores, axis=1)


def evaluation_inputs(
    data: Sales, hierarchy: Hierarchy, origin: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        hierarchy.aggregate(data.actual(origin + 1, origin + HORIZON)),
        revenue_weights(data, hierarchy, origin),
        scale_at(data, hierarchy, origin),
    )


def evaluate(data: Sales, hierarchy: Hierarchy, origin: int, prediction: np.ndarray) -> np.ndarray:
    actual, weight, scale = evaluation_inputs(data, hierarchy, origin)
    return score(actual, prediction, hierarchy, weight, scale)
