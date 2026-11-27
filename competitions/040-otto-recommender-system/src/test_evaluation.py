import pytest
from candidates import METHODS, candidate_aids, rank_features, training_labels
from evaluation import WEEK_MS, Event, recall_counts, split_session, validation_windows


def test_candidate_width_keeps_the_same_leading_items_and_teacher_pool() -> None:
    prefix: list[Event] = [
        {"aid": 123, "ts": 1, "type": "clicks"},
        {"aid": 123, "ts": 2, "type": "carts"},
        {"aid": 124, "ts": 3, "type": "clicks"},
    ]
    popular = list(range(100))
    narrow = candidate_aids(prefix, popular, 40)
    wide = candidate_aids(prefix, popular, 100)
    assert narrow == wide[:40]
    assert wide[:2] == [124, 123]
    assert training_labels(narrow, {124}, METHODS["E01"]) is not None
    assert training_labels(wide, {124}, METHODS["E02"]) is not None
    assert training_labels(wide, {42}, METHODS["E02"]) is None
    assert training_labels(wide, {42}, METHODS["E03"]) is not None


def test_elapsed_feature_uses_last_visible_time_with_missing_sentinel() -> None:
    prefix: list[Event] = [
        {"aid": 7, "ts": 1000, "type": "clicks"},
        {"aid": 7, "ts": 5000, "type": "carts"},
        {"aid": 8, "ts": 6000, "type": "clicks"},
    ]
    rows = rank_features(prefix, [8, 7, 9], {}, elapsed_feature=True)
    assert [row[-1] for row in rows] == [0.0, 1.0, -1.0]


def test_changed_validation_labels_do_not_change_the_prefix_inputs() -> None:
    original: list[Event] = [
        {"aid": 123, "ts": 1, "type": "clicks"},
        {"aid": 124, "ts": 2, "type": "orders"},
    ]
    modified: list[Event] = [original[0], {"aid": 789, "ts": 2, "type": "carts"}]
    first, original_truth = split_session(1, original)
    second, modified_truth = split_session(1, modified)
    assert original_truth != modified_truth
    first_candidates = candidate_aids(first, [789, 4], 20)
    second_candidates = candidate_aids(second, [789, 4], 20)
    assert first_candidates == second_candidates
    assert rank_features(first, first_candidates, {}) == rank_features(
        second, second_candidates, {}
    )


def test_windows_use_half_open_boundaries_and_exclude_crossing_sessions() -> None:
    first, second = validation_windows(2 * WEEK_MS - 1)
    assert first.contains(0, WEEK_MS - 1)
    assert not first.contains(WEEK_MS - 1, WEEK_MS)
    assert second.contains(WEEK_MS, 2 * WEEK_MS - 1)


def test_prefix_and_suffix_labels() -> None:
    events: list[Event] = [
        {"aid": aid, "ts": i, "type": action}
        for i, (aid, action) in enumerate(
            [
                (1, "clicks"),
                (2, "clicks"),
                (3, "clicks"),
                (4, "carts"),
                (4, "carts"),
                (5, "orders"),
                (5, "orders"),
            ]
        )
    ]
    session = next(i for i in range(100) if len(split_session(i, events)[0]) == 1)
    prefix, labels = split_session(session, events)
    assert list(prefix) == events[:1]
    assert labels == {"clicks": {2}, "carts": {4}, "orders": {5}}


def test_recall_caps_denominator_and_retains_empty_targets() -> None:
    truth = {(1, "orders"): set(range(25)), (2, "orders"): set()}
    assert recall_counts(truth, [((1, "orders"), [0, 1]), ((2, "orders"), [0])])["orders"] == (
        2,
        20,
    )
    with pytest.raises(ValueError, match="欠落"):
        recall_counts(truth, [((1, "orders"), [0])])
    with pytest.raises(ValueError, match="商品"):
        recall_counts(truth, [((1, "orders"), [0, 0])])
    with pytest.raises(ValueError, match="重複"):
        recall_counts(truth, [((1, "orders"), [0]), ((1, "orders"), [0])])
