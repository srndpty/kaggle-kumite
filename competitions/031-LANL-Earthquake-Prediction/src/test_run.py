from pathlib import Path

import numpy as np
import pytest
from run import fold_prediction, mae, validate_submission
from windows import fold_ids, rolling_std_quantile, windows

from kumite.checks import check_label_independence, perturb_targets
from kumite.crossfit import folds_to_splits


def test_windows_keep_complete_cycles_and_use_end_labels() -> None:
    y = np.r_[2.0, 1.0, np.linspace(20, 0, 22), np.linspace(20, 0, 12)]
    x = np.arange(len(y), dtype=np.int16)
    chunks = [(x[:2], y[:2]), (x[2:8], y[2:8]), (x[8:24], y[8:24]), (x[24:], y[24:])]
    rows, values, boundaries = windows(chunks, size=10)

    np.testing.assert_array_equal(boundaries[:, 0], [2, 24])
    np.testing.assert_array_equal(rows[:, :3], [[2, 12, 1], [12, 22, 1]])
    np.testing.assert_array_equal(rows[:, 3], y[[11, 21]])
    np.testing.assert_array_equal(values[:, 0], [6.5, 16.5])


def test_fold_isolates_whole_cycles() -> None:
    cycles = np.repeat(np.arange(11), 3)
    folds = fold_ids(cycles)
    np.testing.assert_array_equal(folds[::3], [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    for train, valid in folds_to_splits(folds):
        assert not set(cycles[train]) & set(cycles[valid])


def test_rolling_std_quantile_matches_direct_computation() -> None:
    x = np.array([2.0, 5.0, 8.0, 8.0, 11.0, 13.0])
    width = 3
    expected = np.quantile([x[i : i + width].std() for i in range(len(x) - width + 1)], 0.20)
    assert rolling_std_quantile(x, width) == pytest.approx(expected)


def test_fold_fit_is_independent_of_validation_targets() -> None:
    x = np.arange(80, dtype=float).reshape(-1, 1)
    y = x.ravel() / 10
    train, valid = np.arange(60), np.arange(60, 80)
    changed = np.arange(len(y)) >= 60
    check_label_independence(
        lambda targets: (fold_prediction(x, targets, train, valid),),
        y,
        perturb_targets(y, changed),
    )
    assert mae(np.array([0.0, 1.0]), np.array([1.0, 4.0])) == 2.0


@pytest.mark.parametrize("rows", ["a,0\n", "a,0\na,0\n", "b,0\na,0\n", "a,0\nb,nan\n"])
def test_submission_rejects_invalid_rows(tmp_path: Path, rows: str) -> None:
    path = tmp_path / "submission.csv"
    path.write_text("seg_id,time_to_failure\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission(path, np.array(["a", "b"]))
