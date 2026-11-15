"""特徴量。

- 行ごとに完結する変換（TRANSFORMS）は、どの行に適用しても結果が他の行に依存しない
- ラベルを使う特徴（SupervisedFeature）は、外側 fold ごとに「学習行とそのラベル」だけを受け取って作る
  （呼び出しは pipeline.fold_inputs）。学習行の特徴は内側 fold の OOF、検証行・テスト行の特徴は
  内側の各参照集合での計算の平均にする。外側の検証行のラベルは構造的に読めない
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from cv import make_inner_folds
from dataset import CLASSES, FEATURES
from sklearn.neighbors import NearestNeighbors

from kumite.crossfit import FoldOutput, cross_fit

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def raw(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES]


def log1p(df: pd.DataFrame) -> pd.DataFrame:
    # float32 だと predict_proba も float32 になり、確率の和が 1 からずれて logloss が警告を出す。
    return df[FEATURES].astype("float64").transform(np.log1p)


def row_stats(df: pd.DataFrame) -> pd.DataFrame:
    """生カウントに行合計と非ゼロ数を加える。"""
    x = df[FEATURES]
    return x.assign(row_sum=x.sum(axis=1), row_nnz=(x > 0).sum(axis=1))


def none(df: pd.DataFrame) -> pd.DataFrame:
    """元の特徴を使わない（スタッキングの 2 段目など、ラベルを使う特徴だけで学習する場合）。"""
    return df[[]]


TRANSFORMS: dict[str, Transform] = {
    "raw": raw,
    "log1p": log1p,
    "row_stats": row_stats,
    "none": none,
}


class SupervisedFeature(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def columns(self) -> list[str]: ...

    def __call__(
        self, train: pd.DataFrame, y: pd.Series, query: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """(学習行の特徴（内側 OOF）, query 行の特徴（内側の参照集合ごとの平均）) を返す。"""
        ...


@dataclass(frozen=True)
class KnnDistance:
    """各クラスについて、近傍 n 件までの距離の和（9 クラス × len(neighbors) 列）。

    距離は log1p 空間のユークリッド距離（生カウントのままだと裾の大きい値が距離を支配するため）。
    query 側も内側の各参照集合（学習行の 3/4）で計算して平均し、参照集合の大きさを学習行の特徴と
    そろえる（query だけ近傍が近く出るずれを防ぐ）。
    """

    neighbors: tuple[int, ...] = (1, 2, 4)
    name: str = "knn_dist"

    @property
    def columns(self) -> list[str]:
        return [f"knn_{c}_sum{n}" for c in CLASSES for n in self.neighbors]

    def _distances(self, ref: np.ndarray, ref_y: np.ndarray, query: np.ndarray) -> np.ndarray:
        columns = []
        for k in range(len(CLASSES)):
            nn = NearestNeighbors(n_neighbors=max(self.neighbors), n_jobs=-1).fit(ref[ref_y == k])
            dist, _ = nn.kneighbors(query)
            columns += [dist[:, :n].sum(axis=1) for n in self.neighbors]
        return np.column_stack(columns)

    def __call__(
        self, train: pd.DataFrame, y: pd.Series, query: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        space, space_q, y_arr = log1p(train).to_numpy(), log1p(query).to_numpy(), y.to_numpy()

        def fit_fold(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
            ref, ref_y = space[tr], y_arr[tr]
            return FoldOutput(
                self._distances(ref, ref_y, space[va]), self._distances(ref, ref_y, space_q)
            )

        result = cross_fit(make_inner_folds(y), fit_fold)
        return result.oof, result.query
