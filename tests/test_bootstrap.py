"""グループ単位の bootstrap によるペア比較。"""

import numpy as np
import pytest

from kumite.bootstrap import group_bootstrap_diff

GROUPS = np.array(["a", "a", "b", "b", "c", "c"])


def test_point_is_difference_of_transformed_means() -> None:
    base = np.array([1.0, 1.0, 4.0, 4.0, 9.0, 9.0])
    other = base / 4

    result = group_bootstrap_diff(base, other, GROUPS)

    assert result.point == pytest.approx(np.sqrt(other.mean()) - np.sqrt(base.mean()))
    assert result.low <= result.point <= result.high
    assert result.share_negative == 1.0


def test_identical_losses_give_zero_difference() -> None:
    loss = np.arange(6, dtype=np.float64)

    result = group_bootstrap_diff(loss, loss.copy(), GROUPS)

    assert (result.point, result.low, result.high, result.share_negative) == (0, 0, 0, 0)


def test_same_seed_gives_same_resamples() -> None:
    rng = np.random.default_rng(0)
    base, other = rng.random(6), rng.random(6)

    assert group_bootstrap_diff(base, other, GROUPS, seed=1) == group_bootstrap_diff(
        base, other, GROUPS, seed=1
    )


def test_identity_transform_compares_means() -> None:
    base, other = np.zeros(6), np.ones(6)

    result = group_bootstrap_diff(base, other, GROUPS, transform=lambda m: m)

    assert result.point == pytest.approx(1.0)


def test_rejects_mismatched_shapes_and_nan() -> None:
    with pytest.raises(ValueError):
        group_bootstrap_diff(np.zeros(6), np.zeros(5), GROUPS)
    with pytest.raises(ValueError):
        group_bootstrap_diff(np.full(6, np.nan), np.zeros(6), GROUPS)


def test_rejects_empty_input_missing_groups_and_bad_n_bootstrap() -> None:
    empty = np.zeros(0)
    with pytest.raises(ValueError, match="空"):
        group_bootstrap_diff(empty, empty, np.array([], dtype=object))
    with pytest.raises(ValueError, match="グループ ID"):
        group_bootstrap_diff(np.zeros(2), np.zeros(2), np.array(["a", None], dtype=object))
    with pytest.raises(ValueError, match="n_bootstrap"):
        group_bootstrap_diff(np.zeros(6), np.zeros(6), GROUPS, n_bootstrap=0)
