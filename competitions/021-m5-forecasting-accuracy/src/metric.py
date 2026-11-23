"""公式の12階層を最下層の売上・予測から集計して WRMSSE を計算する。"""

from dataclasses import dataclass

import numpy as np
from dataset import Sales, validate_prediction

from kumite.m5 import LEVELS, historical_change_scale, revenue_weights
from kumite.m5 import Hierarchy as Hierarchy


@dataclass
class Scorer:
    hierarchy: Hierarchy
    weight: np.ndarray
    scale: np.ndarray

    @classmethod
    def at(cls, data: Sales, hierarchy: Hierarchy, origin: int) -> "Scorer":
        weight = revenue_weights(data, hierarchy, origin)
        scale = historical_change_scale(data, hierarchy, origin, squared=True)
        invalid = (weight > 0) & (scale <= 0)
        if invalid.any():
            examples = hierarchy.labels.loc[invalid].head().to_dict("records")
            raise ValueError(f"重みが正の系列で RMSSE の尺度を定義できません: {examples}")
        return cls(hierarchy, weight, scale)

    def score(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        validate_prediction(predicted, self.hierarchy.n_bottom)
        if actual.shape != predicted.shape:
            raise ValueError("実績と予測の形状が一致しません")
        error = self.hierarchy.aggregate(actual - predicted)
        mse = np.mean(np.square(error), axis=1)
        rmsse = np.zeros_like(mse)
        np.divide(mse, self.scale, out=rmsse, where=self.weight > 0)
        return np.bincount(
            self.hierarchy.levels, weights=self.weight * np.sqrt(rmsse), minlength=len(LEVELS)
        )
