"""欠損処理の fold 境界を検証する。"""

import pandas as pd
from train import transform


def test_median_uses_training_rows_only() -> None:
    train = pd.DataFrame({"number": [2.0, 4.0, None]})
    valid = pd.DataFrame({"number": [None, 1000.0]})
    _, result = transform(train, valid, "E02")
    assert result["number"].tolist() == [3.0, 1000.0]
    assert result["number_missing"].tolist() == [True, False]


def test_missing_category_differs_from_unseen_category() -> None:
    train = pd.DataFrame({"category": ["known", None]})
    valid = pd.DataFrame({"category": [None, "unseen", "known"]})
    _, result = transform(train, valid, "E03")
    assert result["category"].cat.codes.tolist() == [1, -1, 0]
