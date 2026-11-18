"""順位と確率を分けて評価する。同点は平均順位（AUC）で扱う。"""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from kumite.checks import validate_probabilities


def normalized_gini(y: np.ndarray, pred: np.ndarray) -> float:
    if y.ndim != 1 or pred.shape != y.shape or len(y) == 0:
        raise ValueError("ラベルと予測は空でない同形の1次元配列が必要です")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("ラベルは 0/1 の両クラスが必要です")
    if not np.isfinite(pred).all():
        raise ValueError("予測に NaN/Inf があります")
    return 2 * float(roc_auc_score(y, pred)) - 1


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """確率を入力とする補助指標一式。Gini 単独は任意の連続スコアを扱える。"""
    gini = normalized_gini(y, pred)
    validate_probabilities(np.column_stack((1 - pred, pred)), len(y), 2)
    return {
        "normalized_gini": gini,
        "auc": float(roc_auc_score(y, pred)),
        "logloss": float(log_loss(y, pred, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, pred)),
    }
