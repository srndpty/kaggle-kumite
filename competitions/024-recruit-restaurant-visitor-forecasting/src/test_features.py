"""予約特徴と祝日で調整した曜日の境界テスト（小さな固定データ）。

実行: uv run pytest competitions/024-recruit-restaurant-visitor-forecasting/src
"""

import numpy as np
import pandas as pd
import pytest
from dataset import DATE, STORE, combine_reserves
from features import (
    DAY,
    FoldContext,
    Reservations,
    holiday_adjusted_dow,
    pseudo_ahead,
    reserved_counts,
)

T = pd.Timestamp
SECOND = pd.Timedelta(seconds=1)


def _reserves(rows: list[tuple[str, str, pd.Timestamp, int]]) -> pd.DataFrame:
    """(店舗, 来店日時, 予約日時, 人数) の並びから予約の表を作る。"""
    return pd.DataFrame(
        {
            STORE: [r[0] for r in rows],
            "visit_datetime": pd.to_datetime([r[1] for r in rows]),
            "reserve_datetime": [r[2] for r in rows],
            "reserve_visitors": [r[3] for r in rows],
            "source": "air",
        }
    )


def _rows(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({STORE: [p[0] for p in pairs], DATE: pd.to_datetime([p[1] for p in pairs])})


# --- reserved_counts: 予約日時による打ち切りと集計 ---


def test_reserved_counts_only_reservations_before_cutoff_for_same_store_and_day() -> None:
    cutoff = T("2017-01-05")
    reserves = _reserves(
        [
            ("a", "2017-01-10 19:00", cutoff - SECOND, 2),  # 起点の 1 秒前: 含める
            ("a", "2017-01-10 12:00", T("2017-01-02 10:00"), 4),  # 同じ日の別の予約: 合算
            ("a", "2017-01-10 18:00", cutoff, 5),  # 起点ちょうど: 含めない
            ("a", "2017-01-11 19:00", T("2017-01-01"), 3),  # 別の来店日: 含めない
            ("b", "2017-01-10 19:00", T("2017-01-01"), 7),  # 別の店舗: 含めない
        ]
    )
    rows = _rows([("a", "2017-01-10"), ("c", "2017-01-10")])

    out = reserved_counts(reserves, rows, pd.Series([cutoff, cutoff]))

    np.testing.assert_array_equal(out, [[6, 2], [0, 0]])  # 予約の無い行は人数 0・件数 0


def test_reserved_uses_each_rows_own_cutoff() -> None:
    reserves = _reserves([("a", "2017-01-10 19:00", T("2017-01-07 12:00"), 3)])
    rows = _rows([("a", "2017-01-10"), ("a", "2017-01-10")])

    out = reserved_counts(reserves, rows, pd.Series([T("2017-01-07"), T("2017-01-08")]))

    np.testing.assert_array_equal(out, [[0, 0], [3, 1]])


def test_combine_reserves_drops_hpg_stores_without_air_id() -> None:
    times = {"visit_datetime": [T("2017-01-10")], "reserve_datetime": [T("2017-01-01")]}
    air = pd.DataFrame({STORE: ["air_1"], **times, "reserve_visitors": [1]})
    hpg = pd.DataFrame(
        {
            "hpg_store_id": ["hpg_1", "hpg_x"],
            "visit_datetime": [T("2017-01-10")] * 2,
            "reserve_datetime": [T("2017-01-01")] * 2,
            "reserve_visitors": [2, 9],
        }
    )
    relation = pd.DataFrame({STORE: ["air_1"], "hpg_store_id": ["hpg_1"]})

    out = combine_reserves(air, hpg, relation)

    assert out[[STORE, "reserve_visitors", "source"]].values.tolist() == [
        ["air_1", 1, "air"],
        ["air_1", 2, "hpg"],
    ]


# --- Reservations.build: 学習・検証・テスト行の打ち切りと「何日先か」 ---


def _context(train: pd.DataFrame, valid: pd.DataFrame, reserves: pd.DataFrame) -> FoldContext:
    origin = T("2017-02-01")
    return FoldContext(
        train=train,
        valid=valid,
        test=_rows([("a", "2017-03-01")]),
        log_y=np.zeros(len(train)),
        reserves=reserves,
        holidays=pd.Series(dtype=np.float64),
        valid_origin=origin,
        test_origin=T("2017-03-01"),
        fit_known_end=T("2017-03-10"),
        test_known_end=T("2017-03-10"),
    )


def test_valid_rows_are_cut_at_origin_with_days_ahead_0_and_38() -> None:
    origin = T("2017-02-01")
    first, last = origin, origin + 38 * DAY
    reserves = _reserves(
        [
            ("a", str(first.date()), origin - SECOND, 1),
            ("a", str(first.date()), origin, 10),
            ("a", str(last.date()), origin - SECOND, 2),
            ("a", str(last.date()), origin, 20),
        ]
    )
    valid = _rows([("a", str(first.date())), ("a", str(last.date()))])

    _, va, _ = Reservations().build(_context(_rows([("a", "2017-01-01")]), valid, reserves))

    np.testing.assert_array_equal(va, [[1, 1, 0], [2, 1, 38]])


def test_valid_rows_all_does_not_cut() -> None:
    origin = T("2017-02-01")
    reserves = _reserves([("a", "2017-02-10", origin + DAY, 5)])
    valid = _rows([("a", "2017-02-10")])
    ctx = _context(_rows([("a", "2017-01-01")]), valid, reserves)

    _, cut, _ = Reservations().build(ctx)
    _, uncut, _ = Reservations(valid_rows="all").build(ctx)

    assert cut[0, 0] == 0 and uncut[0, 0] == 5
    assert cut[0, 2] == uncut[0, 2] == 9  # 何日先かは同じ


def test_train_rows_horizon_cut_matches_days_ahead() -> None:
    dates = pd.date_range("2016-06-01", periods=40, freq="D")
    train = _rows([("a", str(d.date())) for d in dates])
    ahead = pseudo_ahead(len(train))
    cutoffs = dates - pd.to_timedelta(ahead, unit="D")
    # 各学習行に、打ち切り時刻の 1 秒前（含める）とちょうど（含めない）の予約を置く
    reserves = _reserves(
        [("a", str(d.date()), c - SECOND, 1) for d, c in zip(dates, cutoffs, strict=True)]
        + [("a", str(d.date()), c, 100) for d, c in zip(dates, cutoffs, strict=True)]
    )
    ctx = _context(train, _rows([("a", "2017-02-01")]), reserves)

    tr, _, _ = Reservations().build(ctx)

    np.testing.assert_array_equal(tr[:, 0], np.ones(len(train)))
    np.testing.assert_array_equal(tr[:, 2], ahead)


def test_train_rows_all_counts_everything_and_train_ahead_controls_days_column() -> None:
    dates = pd.date_range("2016-06-01", periods=5, freq="D")
    train = _rows([("a", str(d.date())) for d in dates])
    reserves = _reserves([("a", str(d.date()), d + 20 * pd.Timedelta(hours=1), 1) for d in dates])
    ctx = _context(train, _rows([("a", "2017-02-01")]), reserves)

    zero, _, _ = Reservations(train_rows="all").build(ctx)  # E05A
    pseudo, _, _ = Reservations(train_rows="all", train_ahead="pseudo").build(ctx)  # E05B

    np.testing.assert_array_equal(zero[:, :2], np.ones((5, 2)))  # 当日に入った予約も数える
    np.testing.assert_array_equal(pseudo[:, :2], zero[:, :2])
    np.testing.assert_array_equal(zero[:, 2], np.zeros(5))
    np.testing.assert_array_equal(pseudo[:, 2], pseudo_ahead(5))


def test_existing_run_names_are_unchanged() -> None:
    assert Reservations().name == "reservations-train_horizon-valid_cutoff"
    assert Reservations(train_rows="all").name == "reservations-train_all-valid_cutoff"
    assert (
        Reservations(train_rows="all", train_ahead="pseudo").name
        == "reservations-train_all-valid_cutoff-ahead_pseudo"
    )


# --- holiday_adjusted_dow: 祝日を土曜、休日の前日を金曜、翌日を月曜とみなす ---

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
# 祝日の一部（date_info と同じ形: 日付 → フラグ）
HOLIDAYS = pd.Series(
    1,
    index=pd.to_datetime(
        [
            "2016-11-23",
            "2017-01-09",
            "2017-02-11",
            "2017-03-20",
            "2017-04-29",
            "2017-05-03",
            "2017-05-04",
            "2017-05-05",
        ]
    ),
)


def _adjusted(start: str, end: str, known_end: str = "2017-05-31") -> list[int]:
    dates = pd.Series(pd.date_range(start, end, freq="D"))
    return holiday_adjusted_dow(dates, HOLIDAYS, T(known_end)).tolist()


def test_ordinary_week_is_unchanged() -> None:
    # 2017-02-13（月）〜 02-19（日）に祝日は無い
    assert _adjusted("2017-02-13", "2017-02-19") == [MON, TUE, WED, THU, FRI, SAT, SUN]


def test_monday_holiday_becomes_saturday_and_tuesday_becomes_monday() -> None:
    # 2017-01-09（月）成人の日
    assert _adjusted("2017-01-06", "2017-01-11") == [FRI, SAT, SUN, SAT, MON, WED]


def test_midweek_holiday_neighbours_become_friday_and_monday() -> None:
    # 2016-11-23（水）勤労感謝の日: 前日の火曜は金曜、翌日の木曜は月曜とみなす
    assert _adjusted("2016-11-21", "2016-11-25") == [MON, FRI, SAT, MON, FRI]


def test_golden_week_first_middle_and_last_days() -> None:
    # 4/28（金）4/29（土・祝）4/30（日）5/1（月）5/2（火）5/3〜5/5（祝）5/6（土）5/7（日）5/8（月）
    assert _adjusted("2017-04-28", "2017-05-08") == [
        FRI,
        SAT,
        SUN,
        MON,  # 前日が休日、翌日は平日
        FRI,  # 翌日が祝日
        SAT,  # 連休初日
        SAT,  # 中日
        SAT,  # 最終日（翌日は土曜）
        SAT,
        SUN,
        MON,
    ]


def test_holidays_after_known_end_are_unknown() -> None:
    # 予定が 5/2 までしか分からないと、5/3〜5/5 の祝日は見えず、5/2 は普段の火曜のまま
    assert _adjusted("2017-05-01", "2017-05-05", known_end="2017-05-02") == [
        MON,
        TUE,
        WED,
        THU,
        FRI,  # 5/5（金）は暦どおり。翌日が土曜なので金曜のまま
    ]


@pytest.mark.parametrize("known_end", ["2017-05-31", "2017-05-30"])
def test_last_day_of_period_does_not_look_beyond_known_end(known_end: str) -> None:
    # 5/31（水）の翌日 6/1 は予定が分からない範囲。祝日扱いにはならない
    assert _adjusted("2017-05-31", "2017-05-31", known_end=known_end) == [WED]
