import numpy as np
import pytest
from neighbors import knn, normalized, smooth


def _points() -> np.ndarray:
    """1 次元の位置を、内積が近さになる 2 次元の単位ベクトル（角度）にする。"""
    angle = np.array([0.0, 0.1, 0.25, 1.5, 1.6])
    return np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)


def test_knn_finds_nearest_and_excludes_self() -> None:
    idx = knn(_points(), 2)
    np.testing.assert_array_equal(idx[0], [1, 2])
    np.testing.assert_array_equal(idx[3], [4, 2])
    assert not (idx == np.arange(5)[:, None]).any()


def test_knn_breaks_ties_by_row_number() -> None:
    x = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1))  # すべて同じ類似度
    np.testing.assert_array_equal(knn(x, 2), [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]])


@pytest.mark.parametrize("k", [0, -1, 5, 6])
def test_knn_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k は"):
        knn(_points(), k)


def test_knn_rejects_non_finite_features() -> None:
    x = _points()
    x[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        knn(x, 2)


def test_smooth_averages_self_and_neighbours() -> None:
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    idx = np.array([[1], [2], [0]])
    np.testing.assert_allclose(smooth(proba, idx), [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]])


def test_smooth_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="範囲外"):
        smooth(np.full((3, 2), 0.5), np.array([[1], [3], [0]]))


def test_smooth_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="形状"):
        smooth(np.full((3, 2), 0.5), np.array([[1], [0]]))


def test_smooth_rejects_self_in_neighbours() -> None:
    with pytest.raises(ValueError, match="自分自身"):
        smooth(np.full((3, 2), 0.5), np.array([[1], [1], [0]]))


def test_normalized_handles_uniform_rows() -> None:
    x = normalized(np.array([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(x[0], 0.0)
    assert np.linalg.norm(x[1]) == pytest.approx(1.0)
