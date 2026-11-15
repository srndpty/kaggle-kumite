"""検証窓と expanding window 分割。"""

import numpy as np
import pandas as pd
import pytest

from kumite.timesplit import check_time_order, expanding_window_splits, windows


def _panel(n_days: int, series: list[str]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame([(d, s) for d in dates for s in series], columns=["Date", "Series"])


def test_windows_are_contiguous_and_end_at_last_day() -> None:
    result = windows(pd.Timestamp("2020-01-10"), horizon_days=3, n_windows=2)

    assert result == [
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-07")),
        (pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-10")),
    ]


def test_windows_reject_non_positive_sizes() -> None:
    with pytest.raises(ValueError):
        windows(pd.Timestamp("2020-01-10"), horizon_days=0, n_windows=2)


def test_expanding_window_trains_on_all_rows_before_window() -> None:
    panel = _panel(10, ["a", "b"])

    splits = expanding_window_splits(panel["Date"], panel["Series"], ["a", "b"], 3, 2)

    for (tr, va), start in zip(splits, ["2020-01-05", "2020-01-08"], strict=True):
        np.testing.assert_array_equal(tr, np.flatnonzero(panel["Date"] < start))
        assert panel["Date"].iloc[va].min() == pd.Timestamp(start)
        assert len(va) == 3 * 2


def test_expanding_window_validates_only_target_series() -> None:
    panel = _panel(10, ["a", "b"])

    splits = expanding_window_splits(panel["Date"], panel["Series"], ["a"], 3, 2)

    for tr, va in splits:
        assert set(panel["Series"].iloc[va]) == {"a"}
        assert set(panel["Series"].iloc[tr]) == {"a", "b"}  # 学習は全系列


def test_inner_splits_stay_before_last_day() -> None:
    panel = _panel(10, ["a"])
    last_day = pd.Timestamp("2020-01-07")

    splits = expanding_window_splits(panel["Date"], panel["Series"], ["a"], 2, 2, last_day)

    used = np.concatenate([np.concatenate([tr, va]) for tr, va in splits])
    assert panel["Date"].iloc[used].max() == last_day


def test_check_time_order_rejects_overlap() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=4, freq="D"))

    with pytest.raises(ValueError, match="分割 0"):
        check_time_order(dates, [(np.array([0, 2]), np.array([1, 3]))])
