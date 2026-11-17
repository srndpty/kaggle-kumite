"""P2 検証分割。主評価は運転手単位の分割。H1 の比較用に、同じ行を画像単位でランダムに分ける分割も持つ。

test の運転手は学習と別人なので、主評価は「検証 fold の運転手を学習に一切入れない」分割にする。

実行すると分割の診断を表示する:
uv run python competitions/060-state-farm-distracted-driver-detection/src/cv.py
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from dataset import DRIVER, load_train
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from kumite.crossfit import folds_to_splits

SEED = 42
N_FOLDS = 5

FoldMaker = Callable[[pd.DataFrame], np.ndarray]


def driver_folds(train: pd.DataFrame, n_folds: int = N_FOLDS) -> np.ndarray:
    """運転手単位の fold 番号（クラスの比をなるべくそろえる）。入れ子の内側の分割にも使う。"""
    folds = np.full(len(train), -1, dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for k, (_, valid) in enumerate(sgkf.split(train, train["label"], train[DRIVER])):
        folds[valid] = k
    return folds


def random_folds(train: pd.DataFrame) -> np.ndarray:
    """画像単位の層化ランダム fold 番号。同じ運転手のほぼ同じコマが学習と検証の両方に入る。"""
    folds = np.full(len(train), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, valid) in enumerate(skf.split(train, train["label"])):
        folds[valid] = k
    return folds


# 名前 → (split_version, fold 番号の作り方)
SPLITS: dict[str, tuple[str, FoldMaker]] = {
    "driver": (f"driver-sgkf{N_FOLDS}-seed{SEED}-v1", driver_folds),
    "random": (f"image-skf{N_FOLDS}-seed{SEED}-v1", random_folds),
}


def check_driver_disjoint(train: pd.DataFrame, folds: np.ndarray) -> None:
    """各 fold の学習と検証で運転手が重ならないこと。重なれば ValueError。"""
    for k, (tr, va) in enumerate(folds_to_splits(folds)):
        shared = set(train[DRIVER].iloc[tr]) & set(train[DRIVER].iloc[va])
        if shared:
            raise ValueError(f"fold {k} で学習と検証に同じ運転手がいます: {sorted(shared)}")


def main() -> None:
    train = load_train()
    print(f"学習画像 {len(train)} 枚、運転手 {train[DRIVER].nunique()} 人")
    for name, (version, make) in SPLITS.items():
        folds = make(train)
        print(f"\n[{name}] split_version: {version}")
        for k in range(N_FOLDS):
            va = train[folds == k]
            tr_drivers = set(train[DRIVER][folds != k])
            shared = len(set(va[DRIVER]) & tr_drivers)
            counts = np.bincount(va["label"], minlength=10)
            print(
                f"  fold {k}: 検証 {len(va)} 枚、運転手 {va[DRIVER].nunique()} 人"
                f"（学習と共通 {shared} 人）、クラス別の最少 {counts.min()}・最多 {counts.max()}"
            )
    check_driver_disjoint(train, driver_folds(train))
    print("\n運転手単位の分割: 全 fold で学習と検証の運転手が重ならない")


if __name__ == "__main__":
    main()
