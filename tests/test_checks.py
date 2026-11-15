from collections.abc import Callable

import numpy as np
import pytest

from kumite.checks import (
    check_label_independence,
    perturb_targets,
    shift_labels,
    validate_probabilities,
    validate_regression,
)
from kumite.crossfit import FoldOutput, cross_fit

N_CLASSES = 3


def _uniform(n_rows: int) -> np.ndarray:
    return np.full((n_rows, N_CLASSES), 1 / N_CLASSES)


def test_validate_probabilities_accepts_valid() -> None:
    validate_probabilities(_uniform(4), 4, N_CLASSES)


@pytest.mark.parametrize(
    ("value", "message"),
    [(np.nan, "NaN"), (np.inf, "NaN"), (-0.1, "範囲外"), (1.0000001, "範囲外")],
)
def test_validate_probabilities_rejects_invalid_values(value: float, message: str) -> None:
    pred = _uniform(4)
    pred[1, 0] = value
    with pytest.raises(ValueError, match=message):
        validate_probabilities(pred, 4, N_CLASSES)


def test_validate_probabilities_rejects_empty() -> None:
    with pytest.raises(ValueError, match="0 行"):
        validate_probabilities(np.zeros((0, N_CLASSES)), 0, N_CLASSES)


def test_validate_probabilities_rejects_wrong_sum_and_shape() -> None:
    with pytest.raises(ValueError, match="和"):
        validate_probabilities(_uniform(4) * 0.9, 4, N_CLASSES)
    with pytest.raises(ValueError, match="形状"):
        validate_probabilities(_uniform(4), 5, N_CLASSES)


def test_validate_regression_accepts_valid() -> None:
    validate_regression(np.array([0.0, 1.5, 3.0]), 3, lower=0)
    validate_regression(np.array([-1.0, 1.5]), 2)


@pytest.mark.parametrize(
    ("pred", "n_rows", "message"),
    [
        (np.zeros(0), 0, "0 行"),
        (np.zeros((3, 1)), 3, "形状"),
        (np.zeros(2), 3, "形状"),
        (np.array([0.0, np.nan]), 2, "NaN"),
        (np.array([0.0, -np.inf]), 2, "NaN"),
        (np.array([0.0, -0.1]), 2, "下限"),
    ],
)
def test_validate_regression_rejects_invalid(pred: np.ndarray, n_rows: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_regression(pred, n_rows, lower=0)


def test_perturb_targets_changes_only_selected_rows() -> None:
    y = np.array([0, 5, -3, 0, 100, 2])
    rows = np.array([True, False, True, False, True, False])

    perturbed = perturb_targets(y, rows)

    assert (perturbed[rows] != y[rows]).all()
    np.testing.assert_array_equal(perturbed[~rows], y[~rows])
    assert y.dtype == np.int64  # 元の配列は変えない


def test_shift_labels_changes_only_selected_rows() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    rows = np.array([True, False, True, False, True, False])

    shifted = shift_labels(y, rows, N_CLASSES)

    assert (shifted[rows] != y[rows]).all()
    np.testing.assert_array_equal(shifted[~rows], y[~rows])


# ラベルを使う特徴の例: 学習行は内側 fold の OOF、それ以外は学習行全体のクラス比。
Y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
OUTER_FOLDS = np.arange(len(Y)) % 3
HIDDEN = OUTER_FOLDS == 0


def _class_ratio(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=N_CLASSES) / len(y)


def _nested_feature(y: np.ndarray) -> list[np.ndarray]:
    train = np.flatnonzero(~HIDDEN)
    inner_folds = np.arange(len(train)) % 2
    result = cross_fit(
        inner_folds,
        lambda tr, va: FoldOutput(
            np.tile(_class_ratio(y[train][tr]), (len(va), 1)),
            np.tile(_class_ratio(y[train][tr]), (int(HIDDEN.sum()), 1)),
        ),
    )
    return [result.oof, result.query]


def _global_oof_feature(y: np.ndarray) -> list[np.ndarray]:
    """全行で OOF を作ってから外側の学習行を切り出す（隠すべき行のラベルが学習行の特徴に入る）。"""
    result = cross_fit(
        OUTER_FOLDS,
        lambda tr, va: FoldOutput(np.tile(_class_ratio(y[tr]), (len(va), 1)), np.zeros(1)),
    )
    return [result.oof[~HIDDEN]]


def test_nested_feature_does_not_read_hidden_labels() -> None:
    check_label_independence(_nested_feature, Y, shift_labels(Y, HIDDEN, N_CLASSES))


def test_global_oof_feature_is_detected_as_leak() -> None:
    with pytest.raises(AssertionError, match="依存"):
        check_label_independence(_global_oof_feature, Y, shift_labels(Y, HIDDEN, N_CLASSES))


def test_cross_fit_averages_query_over_folds() -> None:
    folds = np.array([0, 1, 0, 1])
    result = cross_fit(
        folds,
        lambda tr, va: FoldOutput(va.astype(float)[:, None], np.array([float(len(tr))])),
    )

    np.testing.assert_array_equal(result.oof[:, 0], [0, 1, 2, 3])
    assert result.query.tolist() == [2.0]
    assert len(result.infos) == 2


@pytest.mark.parametrize(
    "folds",
    [np.array([0, 2]), np.array([0, 0]), np.array([], dtype=int), np.zeros((2, 2), dtype=int)],
)
def test_cross_fit_rejects_invalid_folds(folds: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"folds|fold 番号"):
        cross_fit(folds, lambda tr, va: FoldOutput(np.zeros(len(va)), np.zeros(1)))


def test_cross_fit_rejects_float_folds() -> None:
    with pytest.raises(ValueError, match="整数"):
        cross_fit(np.array([0.0, 1.0]), lambda tr, va: FoldOutput(np.zeros(1), np.zeros(1)))


FOLDS = np.array([0, 1, 0, 1])


def _one_valid_row(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    """valid が 1 行だけ（ブロードキャストで黙って通りうる）。"""
    return FoldOutput(np.zeros((1, 2)), np.zeros(1))


def _valid_width_varies(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    return FoldOutput(np.zeros((len(va), 2 + int(va[0]))), np.zeros(1))


def _query_shape_varies(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    return FoldOutput(np.zeros(len(va)), np.zeros(1 + int(va[0])))


def _nan_valid(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    return FoldOutput(np.full(len(va), np.nan), np.zeros(1))


def _inf_query(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
    return FoldOutput(np.zeros(len(va)), np.array([np.inf]))


@pytest.mark.parametrize(
    ("fit_fold", "message"),
    [
        (_one_valid_row, "行数"),
        (_valid_width_varies, "valid の形"),
        (_query_shape_varies, "query の形"),
        (_nan_valid, "NaN"),
        (_inf_query, "NaN"),
    ],
)
def test_cross_fit_rejects_inconsistent_outputs(
    fit_fold: Callable[[np.ndarray, np.ndarray], FoldOutput], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cross_fit(FOLDS, fit_fold)
