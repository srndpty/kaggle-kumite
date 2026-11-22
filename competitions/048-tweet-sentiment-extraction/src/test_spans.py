"""公式指標、検証分割、原文抽出と提出形式を確認する。"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spans import decode, group_key, jaccard, make_folds, position, write_submission


def test_jaccard() -> None:
    assert jaccard("Yes yes,\tNO", "yes, no") == pytest.approx(2 / 3)


def test_normalized_duplicates_share_fold() -> None:
    text = [f"Text {i}" for i in range(30)]
    frame = pd.DataFrame({"text": [*text, " TEXT   1  "], "sentiment": ["neutral"] * 31})
    folds = make_folds(frame)
    assert set(folds) == set(range(5))
    assert folds[1] == folds[-1]
    assert group_key(text[1]) == group_key(" TEXT   1  ")


def test_span_and_ambiguous_position() -> None:
    text = ' no  "yes,"   please!'
    frame = pd.DataFrame({"text": [text], "sentiment": ["positive"]})
    assert decode(frame, np.array([-1, 0.5, 0.4]), "E01") == ['"yes,"   please!']
    assert decode(frame, np.array([-0.1, -0.2, -0.3]), "E01") == ["no"]
    assert position("yes yes", "yes") is None
    assert position("abc", "bd") is None
    assert position("abc", "bc") == (1, 3)


def test_jaccard_decoder_can_extend_past_half_probability() -> None:
    frame = pd.DataFrame({"text": ["one  two three"], "sentiment": ["positive"]})
    scores = np.array([0.6, -0.2, -0.94])
    assert decode(frame, scores, "E01") == ["one"]
    assert decode(frame, scores, "E05") == ["one  two"]


def test_submission_order_and_quotes(tmp_path: Path) -> None:
    test = pd.DataFrame({"textID": ["b", "a"], "text": ['say "yes," now', "no"]})
    sample = pd.DataFrame({"textID": ["a", "b"]})
    path = tmp_path / "submission.csv"
    write_submission(test, sample, ['"yes,"', "no"], path)
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["textID", "selected_text"],
            ["a", "no"],
            ["b", '"yes,"'],
        ]
    with pytest.raises(ValueError, match="原文"):
        write_submission(test, sample, ["bad", "no"], path)
