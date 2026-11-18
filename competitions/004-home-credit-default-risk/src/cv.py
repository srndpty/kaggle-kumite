"""IDを除く完全重複行を跨がせない層化3fold。"""

import numpy as np
import pandas as pd
from dataset import KEY, unique_ids
from features import application_features
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_folds(main: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    unique_ids(main)
    if len(main) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("行数またはラベルが不正です")
    if min(int((y == 0).sum()), int((y == 1).sum())) < 3:
        raise ValueError("各クラス3行以上が必要です")
    groups = pd.util.hash_pandas_object(application_features(main), index=False).to_numpy()
    grouped = len(np.unique(groups)) < len(groups)
    splitter = (
        StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        if grouped
        else StratifiedKFold(3, shuffle=True, random_state=seed)
    )
    folds = np.full(len(main), -1, dtype=np.int32)
    for k, (tr, va) in enumerate(splitter.split(main[[KEY]], y, groups if grouped else None)):
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError("各foldに両クラスが必要です")
        folds[va] = k
    return folds, f"{'stratified_group3' if grouped else 'stratified3'}-seed{seed}-v1"
