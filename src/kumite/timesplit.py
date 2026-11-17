"""未来を予測する問題の検証分割（予測期間と同じ長さの検証窓を、末尾から並べる）。

test が train の直後の H 日間なら、train の末尾から H 日の窓を n 個とり、各窓を「窓の開始日より前の
全行」で学習して検証する（expanding window）。学習の末尾と検証窓の位置関係が、train と test の関係と
同じになる。
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kumite.crossfit import Split

Window = tuple[pd.Timestamp, pd.Timestamp]  # (開始日, 終了日)。両端を含む


def windows(last_day: pd.Timestamp, horizon_days: int, n_windows: int) -> list[Window]:
    """last_day で終わる、長さ horizon_days の連続した窓を n_windows 個（古い順）。"""
    if horizon_days < 1 or n_windows < 1:
        raise ValueError(f"horizon_days（{horizon_days}）と n_windows（{n_windows}）は 1 以上")
    day = pd.Timedelta(days=1)
    ends = [last_day - k * horizon_days * day for k in reversed(range(n_windows))]
    return [(end - (horizon_days - 1) * day, end) for end in ends]


def window_valid_rows(
    dates: pd.Series,
    series: pd.Series,
    target_series: Sequence[object] | np.ndarray,
    last_day: pd.Timestamp,
    horizon_days: int,
    n_windows: int,
) -> list[np.ndarray]:
    """窓ごとの検証行。target_series（test に出る系列など）に含まれる系列の行だけを検証する。"""
    targeted = series.isin(target_series)
    return [
        np.flatnonzero(targeted & dates.between(start, end))
        for start, end in windows(last_day, horizon_days, n_windows)
    ]


def check_time_order(dates: pd.Series, splits: Sequence[Split]) -> None:
    """各分割で、学習行の最後の日付が検証行の最初の日付より前であることを確かめる。"""
    for k, (tr, va) in enumerate(splits):
        train_end, valid_start = dates.iloc[tr].max(), dates.iloc[va].min()
        if not train_end < valid_start:
            raise ValueError(
                f"分割 {k}: 学習行の末尾 {train_end.date()} が検証行の先頭 {valid_start.date()} より前ではありません"
            )


def expanding_window_splits(
    dates: pd.Series,
    series: pd.Series,
    target_series: Sequence[object] | np.ndarray,
    horizon_days: int,
    n_windows: int,
    last_day: pd.Timestamp | None = None,
) -> list[Split]:
    """各窓を、窓の開始日より前の全行で学習して検証する。

    窓は last_day（既定は dates の最大値）で終わるようにとる。last_day を外側の検証窓の前日にすると、
    外側の学習期間の中だけで閉じた内側の分割になる。
    """
    end: pd.Timestamp = dates.max() if last_day is None else last_day
    starts = [start for start, _ in windows(end, horizon_days, n_windows)]
    valid = window_valid_rows(dates, series, target_series, end, horizon_days, n_windows)
    splits = [
        (np.flatnonzero(dates < start), rows) for start, rows in zip(starts, valid, strict=True)
    ]
    check_time_order(dates, splits)
    return splits
