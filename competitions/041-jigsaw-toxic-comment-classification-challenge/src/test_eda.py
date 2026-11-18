"""重複文面のラベル不一致は診断だけに使い、入力を変更しない。"""

import numpy as np
from cv import text_key
from dataset import LABELS
from eda import label_conflicts


def test_exact_and_normalized_conflicts():
    texts = ["Same", "Same", "same", "ＳＡＭＥ", "Other", "Other", "alone", "steady", "steady"]
    y = np.zeros((len(texts), 6), dtype=int)
    y[1, 0] = 1
    y[5, 1] = 1
    original = y.copy()
    exact = label_conflicts(texts, y)
    normalized = label_conflicts([text_key(t) for t in texts], y)
    assert exact["duplicate_groups"] == normalized["duplicate_groups"] == 3
    assert exact["conflicting_groups"] == normalized["conflicting_groups"] == 2
    assert exact["conflicting_rows"] == 4
    assert normalized["conflicting_rows"] == 6
    assert exact["by_label"][LABELS[0]] == {"groups": 1, "rows": 2}
    assert normalized["by_label"][LABELS[0]] == {"groups": 1, "rows": 4}
    assert normalized["by_label"][LABELS[1]] == {"groups": 1, "rows": 2}
    assert normalized["by_label"][LABELS[2]] == {"groups": 0, "rows": 0}
    np.testing.assert_array_equal(y, original)


def test_no_conflicts():
    result = label_conflicts(["same", "same", "other"], np.zeros((3, 6), dtype=int))
    assert result["duplicate_groups"] == 1
    assert result["conflicting_groups"] == result["conflicting_rows"] == 0
    assert all(v == {"groups": 0, "rows": 0} for v in result["by_label"].values())
