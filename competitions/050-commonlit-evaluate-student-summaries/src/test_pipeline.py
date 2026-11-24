"""課題分割、指標、提出の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from model import write_submission
from train import make_splits, mcrmse

from kumite.crossfit import split_roles


def test_prompt_splits_exclude_duplicates_and_validate_every_row() -> None:
    rows = pd.DataFrame(
        {
            "prompt_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "text": ["Same!", "alpha", "same", "beta", "gamma", "delta", "epsilon", "zeta"],
        }
    )
    splits = make_splits(rows)
    roles = split_roles(len(rows), splits)
    assert ((roles == 2).sum(axis=0) == 1).all()
    for fit, valid in splits:
        assert set(rows.iloc[fit]["prompt_id"]).isdisjoint(rows.iloc[valid]["prompt_id"])
    assert len(splits) == 4
    assert roles[0, 2] == roles[1, 0] == 0


def test_mcrmse_averages_after_square_root() -> None:
    actual = np.zeros((2, 2))
    predicted = np.array([[0, 2], [0, 4]])
    assert mcrmse(actual, predicted) == pytest.approx(np.sqrt(10) / 2)


def test_submission_order_and_values(tmp_path: Path) -> None:
    test = pd.DataFrame({"student_id": ["01", "02"]})
    sample = tmp_path / "template.csv"
    sample.write_text("student_id,content,wording\n02,0,0\n01,0,0\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    write_submission(test, np.array([[1.0, 2.0], [3.0, 4.0]]), output, sample)
    actual = pd.read_csv(output, dtype={"student_id": str})
    assert list(actual) == ["student_id", "content", "wording"]
    assert actual["student_id"].tolist() == ["02", "01"]
    assert actual["content"].tolist() == [3.0, 1.0]
    with pytest.raises(ValueError, match="提出予測"):
        write_submission(test, np.array([[float("inf"), 2.0], [3.0, 4.0]]), output, sample)
