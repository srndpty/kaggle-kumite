"""Kaggle の属性別 AUC と公式合成スコア。"""

from collections.abc import Iterator

import numpy as np
from sklearn.metrics import roc_auc_score

IDENTITIES = (
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
)


def subsets(labels: np.ndarray) -> Iterator[tuple[str, str, np.ndarray]]:
    target = labels[:, 0]
    yield "overall", "overall", np.ones(len(target), dtype=bool)
    for index, identity in enumerate(IDENTITIES, start=1):
        group = labels[:, index]
        yield identity, "subgroup", group
        yield identity, "bpsn", group != target
        yield identity, "bnsp", group == target


def aggregate(aucs: np.ndarray) -> np.ndarray:
    """最終軸に全体 AUC と属性別 AUC を受け取り、4 成分を返す。"""
    values = aucs[..., 1:].reshape((*aucs.shape[:-1], len(IDENTITIES), 3))
    with np.errstate(divide="ignore"):
        bias = np.mean(values**-5, axis=-2) ** -0.2
    return np.concatenate((aucs[..., :1], bias), axis=-1)


def check_support(labels: np.ndarray) -> None:
    for identity, metric, mask in subsets(labels):
        if np.unique(labels[mask, 0]).size != 2:
            raise ValueError(f"{identity} の {metric} に陽性と陰性が必要です")


def breakdown(labels: np.ndarray, prediction: np.ndarray) -> dict:
    rows = []
    for identity, metric, mask in subsets(labels):
        target = labels[mask, 0]
        rows.append(
            {
                "identity": identity,
                "metric": metric,
                "auc": float(roc_auc_score(target, prediction[mask])),
                "positive": int(target.sum()),
                "negative": int(len(target) - target.sum()),
            }
        )
    components = aggregate(np.array([row["auc"] for row in rows]))
    return {
        "score": float(components.mean()),
        "overall_auc": float(components[0]),
        "subgroup_power_mean": float(components[1]),
        "bpsn_power_mean": float(components[2]),
        "bnsp_power_mean": float(components[3]),
        "subsets": rows,
    }


def score(labels: np.ndarray, prediction: np.ndarray) -> float:
    return breakdown(labels, prediction)["score"]
