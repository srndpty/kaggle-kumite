"""明示的な分割（時系列分割など）でのクロスフィット。fold 番号版のテストは test_checks.py にある。"""

import numpy as np
import pytest

from kumite.crossfit import (
    FoldOutput,
    Split,
    cross_fit,
    cross_fit_splits,
    folds_to_splits,
    split_roles,
)

N_ROWS = 6
# 過去で学習して未来を検証する。行 0・1 は一度も検証されず、分割 0 は行 4・5 を使わない。
TIME_SPLITS: list[Split] = [
    (np.array([0, 1]), np.array([2, 3])),
    (np.array([0, 1, 2, 3]), np.array([4, 5])),
]


def _predict_train_size(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    return FoldOutput(np.full(len(va), float(len(tr))), np.array([float(len(tr))]))


def test_cross_fit_splits_leaves_unvalidated_rows_nan() -> None:
    result = cross_fit_splits(N_ROWS, TIME_SPLITS, _predict_train_size)

    np.testing.assert_array_equal(result.oof, [np.nan, np.nan, 2, 2, 4, 4])
    assert result.query.tolist() == [3.0]
    assert len(result.infos) == 2


def test_cross_fit_matches_cross_fit_splits_on_fold_numbers() -> None:
    folds = np.array([0, 1, 2, 0, 1, 2])

    by_folds = cross_fit(folds, _predict_train_size)
    by_splits = cross_fit_splits(len(folds), folds_to_splits(folds), _predict_train_size)

    np.testing.assert_array_equal(by_folds.oof, by_splits.oof)
    np.testing.assert_array_equal(by_folds.query, by_splits.query)


def test_split_roles_marks_train_valid_and_unused() -> None:
    roles = split_roles(N_ROWS, TIME_SPLITS)

    assert roles.tolist() == [[1, 1, 2, 2, 0, 0], [1, 1, 1, 1, 2, 2]]
    assert roles.dtype == np.int8


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ([], "分割数"),
        ([(np.array([0, 1]), np.array([1, 2]))], "重なって"),
        ([(np.array([0]), np.array([1])), (np.array([0]), np.array([1]))], "複数の分割"),
        ([(np.array([0]), np.array([6]))], "範囲外"),
        ([(np.array([0, 0]), np.array([1]))], "重複"),
        ([(np.array([0]), np.array([], dtype=int))], "空でない"),
        ([(np.array([0.0]), np.array([1]))], "整数"),
    ],
)
def test_split_roles_rejects_invalid_splits(splits: list[Split], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        split_roles(N_ROWS, splits)


def test_cross_fit_splits_rejects_invalid_splits_before_fitting() -> None:
    def fail(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
        raise AssertionError("分割が不正なら学習を始めない")

    with pytest.raises(ValueError, match="重なって"):
        cross_fit_splits(N_ROWS, [(np.array([0, 1]), np.array([1, 2]))], fail)
