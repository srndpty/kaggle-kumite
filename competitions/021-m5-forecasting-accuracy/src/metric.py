"""公式の12階層を最下層の売上・予測から集計して WRMSSE を計算する。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dataset import HORIZON, Sales, validate_prediction
from scipy.sparse import csr_matrix

LEVELS: tuple[tuple[str, ...], ...] = (
    (),
    ("state_id",),
    ("store_id",),
    ("cat_id",),
    ("dept_id",),
    ("state_id", "cat_id"),
    ("state_id", "dept_id"),
    ("store_id", "cat_id"),
    ("store_id", "dept_id"),
    ("item_id",),
    ("item_id", "state_id"),
    ("item_id", "store_id"),
)


@dataclass
class Hierarchy:
    matrix: csr_matrix
    levels: np.ndarray
    labels: pd.DataFrame
    n_bottom: int

    @classmethod
    def from_series(cls, series: pd.DataFrame) -> "Hierarchy":
        groups = []
        labels = []
        offset = 0
        for level, keys in enumerate(LEVELS, start=1):
            if keys:
                unique, group = np.unique(
                    series[list(keys)].to_numpy(dtype=str), axis=0, return_inverse=True
                )
                names = ["_".join(row) for row in unique]
            else:
                group = np.zeros(len(series), dtype=np.int64)
                names = ["Total_X"]
            groups.append(group + offset)
            labels.append(pd.DataFrame({"level": level, "series": names}))
            offset += len(names)
        index = np.concatenate(groups)
        columns = np.tile(np.arange(len(series)), len(LEVELS))
        matrix = csr_matrix((np.ones(len(index)), (index, columns)), shape=(offset, len(series)))
        table = pd.concat(labels, ignore_index=True)
        return cls(matrix, table["level"].to_numpy() - 1, table, len(series))

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ values)


@dataclass
class Scorer:
    hierarchy: Hierarchy
    weight: np.ndarray
    scale: np.ndarray

    @classmethod
    def at(cls, data: Sales, hierarchy: Hierarchy, origin: int) -> "Scorer":
        recent = data.actual(origin - HORIZON + 1, origin)
        prices = data.price(origin - HORIZON + 1, origin)
        bad = (recent > 0) & ~np.isfinite(prices)
        if bad.any():
            row, day = np.argwhere(bad)[0]
            raise ValueError(
                f"正の売上に価格がありません: {data.series.iloc[row].to_dict()}, d_{origin - HORIZON + 1 + day}"
            )
        revenue = np.sum(recent * np.nan_to_num(prices, nan=0.0), axis=1, dtype=np.float64)
        dollars = hierarchy.aggregate(revenue)
        total = np.bincount(hierarchy.levels, weights=dollars, minlength=len(LEVELS))
        if (total <= 0).any():
            raise ValueError("売上金額が正の系列がない階層があります")
        weight = dollars / total[hierarchy.levels]
        scale = np.zeros(len(weight))
        # 履歴を階層集計する際の一時配列を128日分に抑える。
        first = np.full(len(weight), origin)
        previous = np.zeros(len(weight))
        for start in range(1, origin + 1, 128):
            end = min(origin, start + 127)
            history = hierarchy.aggregate(data.actual(start, end))
            nonzero = history > 0
            debut = (first == origin) & nonzero.any(axis=1)
            first[debut] = start - 1 + nonzero[debut].argmax(axis=1)
            delta = np.diff(history, axis=1, prepend=previous[:, None])
            delta[np.arange(start - 1, end)[None, :] <= first[:, None]] = 0
            scale += np.einsum("ij,ij->i", delta, delta)
            previous = history[:, -1].copy()
        count = origin - first - 1
        np.divide(scale, count, out=scale, where=count > 0)
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
