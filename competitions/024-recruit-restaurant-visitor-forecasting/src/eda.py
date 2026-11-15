"""P1 EDA。数値は標準出力、図は outputs/eda/ に出す。

実行: uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/eda.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import (
    DATE,
    OUTPUT_DIR,
    STORE,
    TARGET,
    load_air_stores,
    load_date_info,
    load_reserves,
    load_test,
    load_visits,
)
from matplotlib.figure import Figure

FIG_DIR = OUTPUT_DIR / "eda"
COLOR = "#2a6fb0"
HORIZON_DAYS = 39  # test の日数（P0 で実測）
GW_2016 = (pd.Timestamp("2016-04-29"), pd.Timestamp("2016-05-08"))
DOW = ["月", "火", "水", "木", "金", "土", "日"]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


def _with_calendar(df: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """祝日フラグと曜日（月曜 = 0）、「休日」（土日祝）を付ける。"""
    cal = dates.rename(columns={"calendar_date": DATE})[[DATE, "holiday_flg"]]
    out = df.merge(cal, on=DATE, how="left")
    dow = out[DATE].dt.day_of_week
    return out.assign(dow=dow, off=(out["holiday_flg"] == 1) | (dow >= 5))


def overview(visits: pd.DataFrame, test: pd.DataFrame, stores: pd.DataFrame) -> None:
    section("概要")
    for name, df in (("来店記録", visits), ("test", test)):
        days = (df[DATE].max() - df[DATE].min()).days + 1
        print(
            f"{name} {df.shape}: {df[DATE].min().date()} 〜 {df[DATE].max().date()}"
            f"（{days} 日）、店舗 {df[STORE].nunique()}"
        )
        print(f"  (店舗, 日付) の重複: {df.duplicated([STORE, DATE]).sum()}")
    print(f"店舗情報 {stores.shape}。ジャンル {stores['air_genre_name'].nunique()} 種")
    print(f"test の店舗ごとの行数: {test.groupby(STORE).size().describe().to_dict()}")
    print(f"来店記録はあるが test に出ない店舗: {len(set(visits[STORE]) - set(test[STORE]))}")


def coverage(visits: pd.DataFrame) -> None:
    """店舗ごとの記録の始まりと、記録の途中で抜けている日（休業か記録なしか）。"""
    section("店舗ごとの記録の期間と抜け")
    span = visits.groupby(STORE)[DATE].agg(["min", "max", "size"])
    first = span["min"].dt.to_period("M").value_counts().sort_index()
    print(f"記録の始まる月（店舗数）:\n{first.to_string()}")
    print(f"記録の終わる日が train 末尾でない店舗: {(span['max'] < visits[DATE].max()).sum()}")
    days = (span["max"] - span["min"]).dt.days + 1
    missing = days - span["size"]
    print(
        f"記録の期間内で行のない日の割合: {missing.sum() / days.sum():.4f}"
        f"（店舗ごとの中央値 {(missing / days).median():.4f}）"
    )
    # 抜けた日の曜日: 定休日なら特定の曜日に偏る
    present = set(zip(visits[STORE], visits[DATE], strict=True))
    rows = [
        (s, d)
        for s, (lo, hi) in span[["min", "max"]].iterrows()
        for d in pd.date_range(lo, hi)
        if (s, d) not in present
    ]
    gaps = pd.DataFrame(rows, columns=[STORE, DATE])
    share = gaps[DATE].dt.day_of_week.value_counts(normalize=True).sort_index()
    print("抜けた日の曜日の割合:", {DOW[int(k)]: round(float(v), 3) for k, v in share.items()})
    by_month = gaps[DATE].dt.to_period("M").value_counts().sort_index()
    print(f"抜けた日の月ごとの数（年末年始などの一斉休業を見る）:\n{by_month.to_string()}")
    per_store = gaps.groupby(STORE)[DATE].apply(lambda d: d.dt.day_of_week.value_counts().max())
    weekly = per_store / gaps.groupby(STORE).size()
    print(
        f"抜けた日の 50% 以上が同じ曜日の店舗（定休日らしい店舗）: {(weekly >= 0.5).sum()}"
        f" / 抜けのある店舗 {len(weekly)}"
    )


def target_distribution(visits: pd.DataFrame) -> None:
    section("目的変数")
    y = visits[TARGET]
    print(y.describe(percentiles=[0.01, 0.5, 0.99, 0.999]).round(1).to_string())
    log_y = y.apply(np.log1p)
    print(f"歪度: 生 {y.skew():.2f}、log1p {log_y.skew():.2f}")
    store_mean = log_y.groupby(visits[STORE]).transform("mean")
    print(f"log1p 来店数の分散のうち店舗間の差: {store_mean.var() / log_y.var():.3f}")
    by_store = y.groupby(visits[STORE]).median()
    print(
        f"店舗ごとの来店数中央値: 最小 {by_store.min():.0f} / 中央 {by_store.median():.0f}"
        f" / 最大 {by_store.max():.0f}"
    )
    store_median = log_y.groupby(visits[STORE]).transform(
        "median"
    )  # log1p は単調なので中央値と交換できる
    rmsle = float(np.sqrt(((log_y - store_median) ** 2).mean()))
    print(f"参考: 店舗ごとの中央値で予測したときの RMSLE（train 内、未来も見ている）: {rmsle:.4f}")


def calendar_effects(visits: pd.DataFrame) -> None:
    section("曜日・祝日・GW")
    v = visits.assign(log_y=np.log1p(visits[TARGET]))
    print(
        "曜日ごとの来店数の中央値:",
        dict(zip(DOW, v.groupby("dow")[TARGET].median().sort_index().tolist(), strict=True)),
    )
    # 店舗 × 曜日の普段の水準（log の中央値）に対する比
    usual = v.groupby([STORE, "dow"])["log_y"].transform("median")
    v = v.assign(ratio=np.exp(v["log_y"] - usual))
    weekday = v["dow"] < 5
    print(
        f"平日の祝日 / 平日の非祝日: 店舗 × 曜日の普段に対する比の中央値 "
        f"{v.loc[weekday & (v['holiday_flg'] == 1), 'ratio'].median():.3f}"
        f" / {v.loc[weekday & (v['holiday_flg'] == 0), 'ratio'].median():.3f}"
    )
    gw = v[v[DATE].between(*GW_2016)]
    daily = gw.groupby(DATE).agg(ratio=("ratio", "median"), rows=("ratio", "size"))
    daily.index = [f"{d.date()}（{DOW[d.day_of_week]}）" for d in daily.index]
    print(
        f"2016 年 GW の日ごとの比（店舗 × 曜日の普段に対する中央値）:\n{daily.round(3).to_string()}"
    )
    # 休日の前日（金曜・祝前日）: 翌日が休日の平日
    off_by_date = v.drop_duplicates(DATE).set_index(DATE)["off"]
    next_off = v[DATE].map(off_by_date.shift(-1, freq="D")).fillna(False).astype(bool)
    print(
        f"翌日が休日の平日 / それ以外の平日（金曜を除く）: 比の中央値 "
        f"{v.loc[weekday & (v['dow'] < 4) & next_off & ~v['off'], 'ratio'].median():.3f}"
        f" / {v.loc[weekday & (v['dow'] < 4) & ~next_off & ~v['off'], 'ratio'].median():.3f}"
    )


def autocorrelation(visits: pd.DataFrame) -> None:
    section("自己相関（同じ店舗の log1p 来店数。店舗平均を引いた値）")
    wide = visits.assign(log_y=np.log1p(visits[TARGET])).pivot(
        index=DATE, columns=STORE, values="log_y"
    )
    centered = (wide - wide.mean()).asfreq("D").to_numpy()
    for lag in (1, 7, 14, 35, 39, 364):
        a, b = centered[lag:], centered[:-lag]
        ok = np.isfinite(a) & np.isfinite(b)
        corr = np.corrcoef(a[ok], b[ok])[0, 1]
        print(f"  {lag} 日前: {corr:.3f}（{ok.sum()} 組）")


def trend(visits: pd.DataFrame) -> None:
    section("水準の推移")
    month = visits[DATE].dt.to_period("M")
    table = visits.groupby(month).agg(
        stores=(STORE, "nunique"),
        median=(TARGET, "median"),
        mean_log=(TARGET, lambda y: np.log1p(y).mean()),
    )
    print(table.round(3).to_string())
    daily = visits.groupby(DATE).agg(total=(TARGET, "sum"), stores=(STORE, "nunique"))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(daily.index, daily["total"], color=COLOR, lw=0.8)
    ax[0].set_ylabel("来店数の合計")
    ax[1].plot(daily.index, daily["stores"], color=COLOR, lw=0.8)
    ax[1].set_ylabel("記録のある店舗数")
    _save(fig, "daily_total.png")


def reservations(visits: pd.DataFrame, reserves: pd.DataFrame) -> None:
    section("予約")
    r = reserves.assign(
        **{DATE: reserves["visit_datetime"].dt.normalize()},
        lead_days=(
            reserves["visit_datetime"].dt.normalize() - reserves["reserve_datetime"].dt.normalize()
        ).dt.days,
    )
    print(f"予約件数: {r['source'].value_counts().to_dict()}、店舗 {r[STORE].nunique()}")
    print(
        "予約から来店までの日数（暦日）の分布:",
        r["lead_days"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_dict(),
    )
    weights = r["reserve_visitors"]
    for days in (0, 1, 7, 14, 21, 28, 35):
        share = weights[r["lead_days"] >= days].sum() / weights.sum()
        print(f"  来店の {days} 日以上前に入った予約の人数の割合: {share:.3f}")
    per_day = r.groupby([STORE, DATE])["reserve_visitors"].sum().rename("reserved").reset_index()
    joined = visits.merge(per_day, on=[STORE, DATE], how="left")
    has = joined["reserved"].notna()
    with_reserves = joined[STORE].isin(r[STORE].unique())
    print(
        f"来店記録のうち、その日の予約がある行: {has.mean():.3f}"
        f"（予約のある店舗に限ると {has[with_reserves].mean():.3f}）"
    )
    hit = joined[has]
    print(
        f"予約のある行の log1p(予約人数) と log1p(来店数) の相関: "
        f"{np.corrcoef(np.log1p(hit['reserved']), np.log1p(hit[TARGET]))[0, 1]:.3f}"
        f"、予約人数 / 来店数の中央値 {(hit['reserved'] / hit[TARGET]).median():.3f}"
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(r["lead_days"].clip(upper=60), bins=61, weights=weights, color=COLOR)
    ax.set_xlabel("予約から来店までの日数（60 以上はまとめた）")
    ax.set_ylabel("予約人数")
    _save(fig, "reserve_lead_days.png")


def reservation_cutoff(visits: pd.DataFrame, reserves: pd.DataFrame) -> None:
    """test と同じ打ち切り（予測の起点までに入った予約だけ）を、train 末尾の窓で再現する。"""
    section("予約の打ち切り（train 末尾 39 日の窓で再現）")
    end = visits[DATE].max()
    start = end - pd.Timedelta(days=HORIZON_DAYS - 1)
    r = reserves.assign(**{DATE: reserves["visit_datetime"].dt.normalize()})
    in_window = r[r[DATE].between(start, end)]
    known = in_window[in_window["reserve_datetime"] < start]
    print(
        f"窓 {start.date()} 〜 {end.date()} の予約人数: 全部 {in_window['reserve_visitors'].sum()}"
        f"、窓の開始前に入った分 {known['reserve_visitors'].sum()}"
        f"（{known['reserve_visitors'].sum() / in_window['reserve_visitors'].sum():.3f}）"
    )
    day = (known[DATE] - start).dt.days
    full_day = (in_window[DATE] - start).dt.days
    share = (
        known.groupby(day)["reserve_visitors"].sum()
        / in_window.groupby(full_day)["reserve_visitors"].sum()
    ).reindex(range(HORIZON_DAYS))
    print("窓の開始からの日数ごとの、開始前に入っていた予約人数の割合（7 日ごとの平均）:")
    print(
        {
            f"{k}〜{k + 6}": round(float(share.iloc[k : k + 7].mean()), 3)
            for k in range(0, HORIZON_DAYS, 7)
        }
    )
    test_start = end + pd.Timedelta(days=1)
    test_r = r[r[DATE] >= test_start]
    test_day = (test_r[DATE] - test_start).dt.days
    print(
        "test の日数ごとの予約人数の合計（7 日ごと）:",
        {
            f"{k}〜{k + 6}": int(test_r.loc[test_day.between(k, k + 6), "reserve_visitors"].sum())
            for k in range(0, HORIZON_DAYS, 7)
        },
    )
    print(
        "参考: 窓の開始前に入っていた予約人数（7 日ごと）:",
        {
            f"{k}〜{k + 6}": int(known.loc[day.between(k, k + 6), "reserve_visitors"].sum())
            for k in range(0, HORIZON_DAYS, 7)
        },
    )


def train_test_shift(visits: pd.DataFrame, test: pd.DataFrame, dates: pd.DataFrame) -> None:
    section("train と test の分布差")
    end = visits[DATE].max()
    last = visits[visits[DATE] > end - pd.Timedelta(days=HORIZON_DAYS)]
    year_ago = visits[
        visits[DATE].between(
            test[DATE].min() - pd.DateOffset(years=1), test[DATE].max() - pd.DateOffset(years=1)
        )
    ]
    rows = {}
    for name, df in (
        ("test", test),
        ("train 末尾 39 日", last),
        ("1 年前の同時期", year_ago),
        ("train 全体", visits),
    ):
        c = _with_calendar(df[[STORE, DATE]], dates)
        rows[name] = {
            "店舗": df[STORE].nunique(),
            "祝日": round(float((c["holiday_flg"] == 1).mean()), 3),
            "土日祝": round(float(c["off"].mean()), 3),
        }
    print(pd.DataFrame(rows).T.to_string())
    print(
        f"1 年前の同時期に記録のある、test の店舗: "
        f"{test[STORE].isin(year_ago[STORE]).groupby(test[STORE]).first().sum()} / {test[STORE].nunique()}"
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
    pd.set_option("display.width", 160)
    dates = load_date_info()
    visits = _with_calendar(load_visits(), dates)
    test, stores, reserves = load_test(), load_air_stores(), load_reserves()
    overview(visits, test, stores)
    coverage(visits)
    target_distribution(visits)
    calendar_effects(visits)
    autocorrelation(visits)
    trend(visits)
    reservations(visits, reserves)
    reservation_cutoff(visits, reserves)
    train_test_shift(visits, test, dates)


if __name__ == "__main__":
    main()
