from data import Pair, make_folds, parse_location
from model import decode, decode_pair


def test_duplicate_note_text_shares_fold() -> None:
    pairs = [
        Pair(str(index), "0", str(index), "0", f"note {index // 2}", "feature", [])
        for index in range(20)
    ]
    pairs.append(Pair("extra", "0", "0", "1", "note 0", "other", []))
    folds = make_folds(pairs)
    assert set(folds) == set(range(5))
    assert folds[0] == folds[-1]
    assert all(folds[index] == folds[index + 1] for index in range(0, 20, 2))


def test_multiple_locations_and_whitespace_bridging() -> None:
    text = "a \nb,c"
    assert parse_location("['0 1;3 4', '5 6']", len(text)) == [(0, 1), (3, 4), (5, 6)]
    scores = {(0, 1): 0.9, (3, 4): 0.5, (5, 6): 0.9}
    assert decode(text, scores, join_whitespace=False) == [(0, 1), (3, 4), (5, 6)]
    assert decode(text, scores, join_whitespace=True) == [(0, 4), (5, 6)]


def test_duration_rule_only_filters_its_feature() -> None:
    text = "2 weeks, 2 months, years"
    scores = {(0, 1): 0.9, (2, 7): 0.9, (9, 10): 0.9, (11, 17): 0.9, (19, 24): 0.9}
    target = Pair("a", "3", "1", "309", text, "duration", [])
    other = Pair("b", "3", "1", "308", text, "other", [])
    assert decode_pair(target, scores, "E04") == [(9, 10), (11, 17)]
    assert decode_pair(other, scores, "E04") == list(scores)
    assert decode_pair(target, scores, "B00") == list(scores)
