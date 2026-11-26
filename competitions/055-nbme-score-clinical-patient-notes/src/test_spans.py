import pytest
from spans import micro_f1, normalize


def test_official_example() -> None:
    assert micro_f1([(10, [(0, 3), (3, 5)], [(2, 5), (7, 9), (2, 3)])]) == 0.6


def test_aggregate_counts_and_empty_prediction() -> None:
    assert micro_f1([(10, [(0, 1)], [(0, 1)]), (10, [(0, 3)], [])]) == 0.4
    assert micro_f1([(10, [], [])]) == 0.0


def test_offsets_use_characters() -> None:
    text = "a\n🦉z"
    assert text[slice(*normalize([(2, 3)], len(text))[0])] == "🦉"
    with pytest.raises(ValueError, match="範囲外"):
        normalize([(2, 5)], len(text))
