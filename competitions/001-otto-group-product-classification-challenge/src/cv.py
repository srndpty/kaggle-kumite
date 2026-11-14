"""P2 検証分割。全実験でこの分割を使い、同一分割でペア比較する。

実行すると分割の診断（fold ごとの件数・クラス比）を表示する:
uv run python competitions/001-otto-group-product-classification-challenge/src/cv.py
"""

import numpy as np
import pandas as pd
from dataset import CLASSES, TARGET, encode_target, load_train
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
SEED = 42
SPLIT_VERSION = f"skf{N_SPLITS}-seed{SEED}"


def make_folds(y: pd.Series) -> np.ndarray:
    """各行の検証 fold 番号 (0..N_SPLITS-1) を返す。train は class 順に並ぶので shuffle 必須。"""
    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    return folds


def main() -> None:
    y = encode_target(load_train()[TARGET])
    folds = make_folds(y)
    print(f"split_version: {SPLIT_VERSION}")
    table = pd.crosstab(folds, y, normalize="index").round(4)
    table.columns = CLASSES
    table.insert(0, "n_rows", np.bincount(folds))
    print(table.to_string())
    print(f"\nfold 間のクラス比の最大差: {(table[CLASSES].max() - table[CLASSES].min()).max():.4f}")


if __name__ == "__main__":
    main()
