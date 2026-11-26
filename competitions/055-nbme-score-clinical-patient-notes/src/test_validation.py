import json
from pathlib import Path

import pytest
from data import Pair, index_rows, make_folds, parse_location
from train import read_probabilities


def test_reject_invalid_label_and_duplicate_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="文字列のリスト"):
        parse_location('"0 1"', 5)
    path = tmp_path / "notes.csv"
    path.write_text("case_num,pn_num,pn_history\n0,1,a\n0,1,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="結合キーが重複"):
        index_rows(path, "pn_num", "pn_history")
    with pytest.raises(ValueError, match="ケースまたは本文が一致しません"):
        make_folds([Pair("a", "0", "1", "1", "x", "f", []), Pair("b", "1", "1", "1", "x", "f", [])])


@pytest.mark.parametrize(
    "scores",
    [
        [(0, 1, 0.1), (0, 1, 0.2)],
        [(0, 3, 0.1)],
        [(0, 1, float("nan"))],
        [(0, 1, 1.1)],
        [(False, 1, 0.1)],
    ],
)
def test_reject_bad_baseline_token_scores(tmp_path: Path, scores: list[object]) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps([{"id": "a", "scores": scores}]), encoding="utf-8")
    with pytest.raises(ValueError, match="範囲または確率が不正"):
        read_probabilities(path, [Pair("a", "0", "1", "1", "xy", "f", [])])
