"""選定時に実験ログと保存済み予測を再照合する。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from select_model import validate_oof
from spans import training_sha256

from kumite.experiments import ExperimentRecord


@pytest.mark.parametrize("change", ["log", "prediction", "fold"])
def test_reject_inconsistent_oof(change: str) -> None:
    train = pd.DataFrame({"textID": ["a"], "selected_text": ["one"]})
    oof = pd.DataFrame({"textID": ["a"], "fold": [0], "prediction": ["one"], "score": [1.0]})
    record = ExperimentRecord(
        competition_id="048",
        run_id="E02",
        metric="word Jaccard",
        greater_is_better=True,
        data_version="official-v1",
        data_hash="",
        split_version="test",
        split_hash="",
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="",
        fold_scores=[1.0],
        oof_score=1.0,
        code_version="test",
    )
    folds = np.array([0])
    np.testing.assert_array_equal(validate_oof(oof, train, folds, record), [1.0])
    if change == "log":
        record = replace(record, oof_score=0.5)
        message = "実験ログの OOF スコア"
    elif change == "prediction":
        oof.loc[0, "prediction"] = "one two"
        message = "行スコア"
    else:
        oof.loc[0, "fold"] = 1
        message = "fold"
    with pytest.raises(ValueError, match=message):
        validate_oof(oof, train, folds, record)


@pytest.mark.parametrize("column", ["textID", "text", "sentiment", "selected_text"])
def test_training_hash_includes_all_used_columns_and_order(column: str) -> None:
    train = pd.DataFrame(
        {
            "textID": ["a", "b"],
            "text": ["  café\nYes", "no"],
            "sentiment": ["positive", "negative"],
            "selected_text": ["café", "no"],
        }
    )
    digest = training_sha256(train)
    assert training_sha256(train[train.columns[::-1]]) == digest
    assert training_sha256(train.iloc[::-1]) != digest
    train.loc[0, column] = str(train.loc[0, column]) + " "
    assert training_sha256(train) != digest
