from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ranking import KEY, LABEL, read_queries, retrieve, submission
from train import map25, splits_for

from kumite.checks import check_label_independence


def test_map25() -> None:
    ranking = np.tile(np.arange(100), (3, 1))
    assert map25(np.array([0, 24, 25]), ranking) == pytest.approx((1 + 1 / 25) / 3)


def test_retrieval_is_independent_of_labels(tmp_path: Path) -> None:
    queries = pd.DataFrame({"text": ["subtract three", "add three"]})
    names = ["subtract", "addition", "multiply"]

    def build(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return retrieve(queries.assign(**{LABEL: labels}), names, "E00", tmp_path)

    check_label_independence(build, np.array([0, 1]), np.array([0, 2]))


def test_expansion_and_duplicate_groups(tmp_path: Path) -> None:
    rows = [
        {
            "QuestionId": i,
            "QuestionText": "  SAME question " if i == 1 else "same question" if i == 0 else str(i),
            "SubjectName": "Math",
            "ConstructName": "Topic",
            "CorrectAnswer": "A",
            **{f"Answer{letter}Text": letter for letter in "ABCD"},
            **{f"Misconception{letter}Id": 3 if letter in "AB" else "" for letter in "ABCD"},
        }
        for i in range(6)
    ]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = read_queries(path, labeled=True)
    assert frame[KEY].tolist() == [f"{i}_B" for i in range(6)]
    assert len(read_queries(path, labeled=False)) == 18
    folds = splits_for(frame)
    assert sorted(np.concatenate([valid for _, valid in folds])) == list(range(6))
    for train, valid in folds:
        assert set(frame.iloc[train]["group"]).isdisjoint(frame.iloc[valid]["group"])


def test_submission_validates_ids_and_uses_sample_order() -> None:
    ids = np.arange(30)
    queries = pd.DataFrame({KEY: ["4_B", "4_C"]})
    sample = pd.DataFrame({KEY: ["4_C", "4_B"]})
    pred = np.stack((ids[:25], ids[1:26]))
    result = submission(sample, queries, pred, ids)
    assert result[KEY].tolist() == ["4_C", "4_B"]
    assert result[LABEL].iloc[0] == " ".join(map(str, ids[1:26]))
    with pytest.raises(ValueError, match="ID"):
        submission(sample.iloc[:1], queries, pred, ids)
    pred[0, 0] = 1
    with pytest.raises(ValueError, match="重複"):
        submission(sample, queries, pred, ids)
