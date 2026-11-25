import numpy as np
import pytest

from kumite.ranking import average_precision, sparse_hits, validate_full_rankings


def test_average_precision_by_hand() -> None:
    hits = np.array([[True, False, True], [False, False, False], [True, True, True]])
    # 正解 2 件 / 正解 0 件 / 正解が K より多い
    np.testing.assert_allclose(
        average_precision(hits, np.array([2, 0, 9])), [(1 + 2 / 3) / 2, 0.0, 1.0]
    )


def test_average_precision_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="形"):
        average_precision(np.zeros((2, 3), dtype=bool), np.zeros(3))


def test_sparse_hits_do_not_cross_queries() -> None:
    ranked = np.array([[1, 4], [1, -1]])
    # 2 行目の -1 は 5 * 1 - 1 = 4 となり、1 行目の商品 4 と衝突しうる
    hits = sparse_hits(ranked, np.array([0, 1]), np.array([4, 1]), 5)
    assert hits.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    "ranked", [np.array([[0, 0]]), np.array([[0, 5]]), np.array([[-1, 0]]), np.array([[0.0, 1.0]])]
)
def test_validate_full_rankings_rejects_invalid(ranked: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_full_rankings(ranked, (1, 2), 5)


def test_validate_full_rankings_accepts_valid() -> None:
    validate_full_rankings(np.array([[4, 0]]), (1, 2), 5)
