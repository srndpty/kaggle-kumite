"""閾値と分割のリーク防止を確かめる。"""

import numpy as np
import pandas as pd
from model import f1_threshold, group_keys, predict, splits

from kumite.checks import check_label_independence


def test_threshold_uses_whole_ties_and_prefers_highest() -> None:
    assert f1_threshold(np.array([1, 0, 0, 1]), np.array([0.8, 0.3, 0.3, 0.3])) == 0.8
    assert f1_threshold(np.array([0, 1, 1]), np.array([0.9, 0.6, 0.6])) == 0.6


def test_outer_labels_do_not_change_model_or_threshold() -> None:
    text = pd.Series(
        [f"{'apple' if i % 2 else 'orange'} fruit {i}" for i in range(36)]
        + ["ＡＰＰＬＥ   fruit 1"]
    )
    y = np.arange(len(text)) % 2
    groups = group_keys(text)
    assert groups[1] == groups[-1]
    pool, valid = splits(y, groups)[0]
    fit, tune = splits(y[pool], groups[pool])[0]
    assert not set(groups[pool]) & set(groups[valid])
    assert not set(groups[pool[fit]]) & set(groups[pool[tune]])
    docs = text.to_numpy()

    def build(labels: np.ndarray) -> tuple[np.ndarray, ...]:
        pt, pv, coverage = predict(docs, labels, pool[fit], pool[tune], docs[valid])
        threshold = f1_threshold(labels[pool[tune]], pt)
        return pt, pv, coverage, np.array([threshold]), pv >= threshold

    changed = y.copy()
    changed[valid] = 1 - y[valid]
    check_label_independence(build, y, changed, atol=0)
