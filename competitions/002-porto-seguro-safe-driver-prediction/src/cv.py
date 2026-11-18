"""完全重複の特徴行を跨がせない、固定した層化5分割。"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SEED = 42


def make_folds(x: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, str]:
    """完全一致する特徴行だけをグループ化する。

    特徴が異なる同一人物・契約など、潜在的なグループを検出するものではない。
    """
    if len(y) != len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError("行数またはラベルが不正です")
    if min(np.count_nonzero(y == 0), np.count_nonzero(y == 1)) < 5:
        raise ValueError("5分割には各クラス5行以上が必要です")
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    grouped = len(np.unique(groups)) != len(groups)
    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        if grouped
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    folds = np.full(len(y), -1, dtype=np.int32)
    for k, (_, va) in enumerate(splitter.split(x, y, groups if grouped else None)):
        if (
            len(np.unique(y[va])) != 2
            or len(np.unique(y[np.setdiff1d(np.arange(len(y)), va)])) != 2
        ):
            raise ValueError("各 fold の学習・検証には両クラスが必要です")
        folds[va] = k
    name = "stratified_group5" if grouped else "stratified5"
    return folds, f"{name}-seed{seed}-v1"
