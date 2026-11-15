"""nested stacking: 1 段目を外側 fold の学習行だけで学習し直し、その予測確率を 2 段目の特徴にする。

保存済みの OOF（全行で作ったもの）を 2 段目に使うと、外側の検証行のラベルが 1 段目のモデルを通して
2 段目の学習行の特徴に入る。ここでは 1 段目を毎回、外側の学習行の内側 fold で作り直してこれを防ぐ。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from cv import make_inner_folds
from dataset import CLASSES
from pipeline import Experiment, run_cv


@dataclass(frozen=True)
class Stacking:
    base: tuple[tuple[str, Experiment], ...]  # (run_id, 1 段目の実験定義)

    @property
    def name(self) -> str:
        return f"stack({','.join(run_id for run_id, _ in self.base)})"

    @property
    def columns(self) -> list[str]:
        return [f"{run_id}_{c}" for run_id, _ in self.base for c in CLASSES]

    def __call__(
        self, train: pd.DataFrame, y: pd.Series, query: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        folds = make_inner_folds(y)
        oofs, queries = [], []
        for run_id, exp in self.base:
            result = run_cv(exp, train, y, folds, query)
            oofs.append(result.oof)
            queries.append(result.query)
            print(f"  1 段目 {run_id} 完了", flush=True)
        return np.hstack(oofs), np.hstack(queries)
