"""公式指標と補助指標。"""

from dataset import LABELS
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def macro_auc(y, pred) -> float:
    return float(roc_auc_score(y, pred, average="macro"))


def scores(y, pred):
    return {
        "macro_auc": macro_auc(y, pred),
        "labels": {
            label: {
                "auc": float(roc_auc_score(y[:, i], pred[:, i])),
                "logloss": float(log_loss(y[:, i], pred[:, i], labels=[0, 1])),
                "ap": float(average_precision_score(y[:, i], pred[:, i])),
            }
            for i, label in enumerate(LABELS)
        },
    }
