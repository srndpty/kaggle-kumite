"""P1 EDA。数値は標準出力、図は outputs/eda/ に出す。

実行: uv run python competitions/019-rossmann-store-sales/src/eda.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import OUTPUT_DIR, TARGET, load_store, load_test, load_train
from features import GAP_COLUMNS, Calendar, Gaps
from matplotlib.figure import Figure

FIG_DIR = OUTPUT_DIR / "eda"
COLOR = "#2a6fb0"  # 図は 1 系列なので 1 色に固定する


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


def overview(train: pd.DataFrame, test: pd.DataFrame, store: pd.DataFrame) -> None:
    section("概要")
    for name, df in (("train", train), ("test", test)):
        days = (df["Date"].max() - df["Date"].min()).days + 1
        print(
            f"{name} {df.shape}: {df['Date'].min().date()} 〜 {df['Date'].max().date()}"
            f"（{days} 日）、店舗 {df['Store'].nunique()}"
        )
        print(f"  欠損: {df.isna().sum()[lambda s: s > 0].to_dict()}")
        print(f"  (Store, Date) の重複: {df.duplicated(['Store', 'Date']).sum()}")
    print(f"store {store.shape}。欠損: {store.isna().sum()[lambda s: s > 0].to_dict()}")
    test_stores = set(test["Store"])
    print(f"test の店舗のうち train にない店舗: {len(test_stores - set(train['Store']))}")
    print(f"test の店舗ごとの行数: {test.groupby('Store').size().describe().to_dict()}")
    print(f"train の DayOfWeek と日付の曜日（月曜=1）が一致: {_dow_matches(train)}")


def _dow_matches(df: pd.DataFrame) -> bool:
    return bool((df["Date"].dt.day_of_week + 1 == df["DayOfWeek"]).all())


def open_and_zero_sales(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("休業日と売上 0")
    print(pd.crosstab(train["Open"], train[TARGET] > 0, rownames=["Open"], colnames=["Sales>0"]))
    print(f"train の Open=0 の割合: {(train['Open'] == 0).mean():.4f}")
    print(f"test の Open: {test['Open'].value_counts(dropna=False).to_dict()}")
    missing = test[test["Open"].isna()]
    print(
        f"test の Open 欠損: 店舗 {sorted(missing['Store'].unique())}、"
        f"{missing['Date'].min().date()} 〜 {missing['Date'].max().date()}、"
        f"曜日 {sorted(missing['DayOfWeek'].unique())}"
    )
    print(
        "曜日ごとの Open=0 の割合（train）:",
        train.groupby("DayOfWeek")["Open"]
        .apply(lambda s: round(float((s == 0).mean()), 4))
        .to_dict(),
    )


def target_distribution(train: pd.DataFrame) -> None:
    section("目的変数（営業して売上のある行）")
    sales = train.loc[train[TARGET] > 0, TARGET]
    print(sales.describe(percentiles=[0.01, 0.5, 0.99, 0.999]).round(1).to_string())
    log_sales = pd.Series(np.log1p(sales.to_numpy()), index=sales.index)
    print(f"歪度: 生 {sales.skew():.2f}、log1p {log_sales.skew():.2f}")
    by_store = sales.groupby(train["Store"]).median()
    print(
        f"店舗ごとの売上中央値: 最小 {by_store.min():.0f} / 中央 {by_store.median():.0f} / 最大 {by_store.max():.0f}"
    )
    # 売上の変動のうち、店舗の水準で説明できる割合（log1p 空間）
    between = log_sales.groupby(train["Store"]).transform("mean")
    print(f"log1p 売上の分散のうち店舗間の割合: {between.var() / log_sales.var():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(sales, bins=100, color=COLOR)
    axes[0].set_title("Sales (open days, Sales > 0)")
    axes[1].hist(log_sales, bins=100, color=COLOR)
    axes[1].set_title("log1p(Sales)")
    _save(fig, "sales_distribution.png")


def calendar_effects(train: pd.DataFrame) -> None:
    section("曜日・販促・祝日・月の効果（売上 > 0 の行の中央値）")
    sold = train[train[TARGET] > 0]
    for col in ("DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"):
        table = sold.groupby(col)[TARGET].agg(["median", "size"])
        print(f"{col}:\n{table.to_string()}")
    month = sold.groupby(sold["Date"].dt.month)[TARGET].median()
    print(f"月:\n{month.to_string()}")
    promo_ratio = sold.groupby(["Store", "Promo"])[TARGET].median().unstack()
    ratio = (promo_ratio[1] / promo_ratio[0]).describe().round(3)
    print(f"店舗ごとの「販促あり / なし」の売上中央値の比: {ratio.to_dict()}")
    print(
        f"Promo は同じ日の全店舗で同じ値か: {bool(train.groupby('Date')['Promo'].nunique().le(1).all())}"
    )

    daily = sold.groupby("Date")[TARGET].median()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(daily.index, daily.to_numpy(), color=COLOR, linewidth=1)
    ax.set_title("median Sales per day (stores with Sales > 0)")
    _save(fig, "daily_median_sales.png")


def store_coverage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("店舗ごとの欠測期間")
    n_days = train["Date"].nunique()
    rows = train.groupby("Store").size()
    print(f"train の日数: {n_days}。全日そろっている店舗: {(rows == n_days).sum()}")
    print(f"行数の分布: {rows.value_counts().head(5).to_dict()}")
    partial = rows[rows < n_days].index
    per_day = train.groupby("Date")["Store"].nunique()
    low = per_day[per_day < per_day.max()]
    print(
        f"店舗数が最大（{per_day.max()}）に満たない日: {len(low)} 日、"
        f"{low.index.min().date()} 〜 {low.index.max().date()}、その間の店舗数 {low.min()}〜{low.max()}"
    )
    in_test = np.isin(partial, test["Store"].unique()).sum()
    print(f"欠測のある店舗 {len(partial)} のうち test に出る店舗: {in_test}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(per_day.index, per_day.to_numpy(), color=COLOR, linewidth=1.5)
    ax.set_ylim(0, None)
    ax.set_title("number of stores with a row per day (train)")
    _save(fig, "stores_per_day.png")


def store_attributes(train: pd.DataFrame, store: pd.DataFrame) -> None:
    section("店舗属性")
    sold = train[train[TARGET] > 0].merge(store, on="Store")
    for col in ("StoreType", "Assortment", "Promo2"):
        table = sold.groupby(col).agg(
            median_sales=(TARGET, "median"), stores=("Store", "nunique"), rows=(TARGET, "size")
        )
        print(f"{col}:\n{table.to_string()}")
    print(f"PromoInterval: {store['PromoInterval'].value_counts(dropna=False).to_dict()}")
    print(f"CompetitionDistance: {store['CompetitionDistance'].describe().round(0).to_dict()}")
    years = store["CompetitionOpenSinceYear"]
    print(
        f"CompetitionOpenSinceYear: 最小 {years.min()}、最大 {years.max()}、欠損 {years.isna().sum()}"
    )
    print(
        f"Promo2=0 の店舗で Promo2Since が欠損: {store.loc[store['Promo2'] == 0, 'Promo2SinceYear'].isna().all()}"
    )
    by_store = sold.groupby("Store").agg(
        sales=(TARGET, "median"), dist=("CompetitionDistance", "first")
    )
    print(
        f"Spearman(店舗の売上中央値, 競合店までの距離): {by_store['sales'].corr(by_store['dist'], method='spearman'):.3f}"
    )


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("train / test の分布差（test と同じ長さの train 末尾・1 年前の同時期と比べる）")
    length = (test["Date"].max() - test["Date"].min()).days + 1
    end = train["Date"].max()
    tail = train[train["Date"] > end - pd.Timedelta(days=length)]
    year_ago = train[
        train["Date"].between(
            test["Date"].min() - pd.DateOffset(years=1), test["Date"].max() - pd.DateOffset(years=1)
        )
    ]
    for name, df in (
        ("test", test),
        ("train 末尾", tail),
        ("1 年前の同時期", year_ago),
        ("train 全体", train),
    ):
        print(
            f"{name}: 行 {len(df)}、店舗 {df['Store'].nunique()}、Open=0 {(df['Open'] == 0).mean():.4f}、"
            f"Promo {df['Promo'].mean():.4f}、SchoolHoliday {df['SchoolHoliday'].mean():.4f}、"
            f"StateHoliday≠0 {(df['StateHoliday'] != '0').mean():.4f}"
        )


def time_structure(train: pd.DataFrame) -> None:
    section("時間方向の構造（リークと分割の検討用）")
    sold = train[train[TARGET] > 0].copy()
    sold["log_sales"] = np.log1p(sold[TARGET])
    wide = sold.pivot(index="Date", columns="Store", values="log_sales")
    for lag in (1, 7, 14, 49, 364):
        corr = wide.corrwith(wide.shift(lag, freq="D")).median()
        print(f"同じ店舗の {lag} 日前の log 売上との相関（店舗の中央値）: {corr:.3f}")
    yearly = sold.groupby(sold["Date"].dt.year)[TARGET].median()
    print(f"年ごとの売上中央値: {yearly.to_dict()}")
    same_months = sold[sold["Date"].dt.month <= 7]
    print(
        f"1〜7 月だけの年ごとの売上中央値: {same_months.groupby(same_months['Date'].dt.year)[TARGET].median().to_dict()}"
    )


def closure_effect(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("休業日・祝日の前後（売上 / 店舗 × 曜日 × 販促の中央値。1 なら普段どおり）")
    calendar = Calendar.build(train, test)
    gaps = calendar.gaps(Gaps(), calendar.end)
    sold = train[train[TARGET] > 0].merge(gaps, on=["Store", "Date"])
    usual = sold.groupby(["Store", "DayOfWeek", "Promo"])[TARGET].transform("median")
    sold = sold.assign(ratio=sold[TARGET] / usual)
    # 日曜定休の店は土曜・月曜が毎週「休業の前後」になるので、火〜金だけで見る
    midweek = sold[sold["DayOfWeek"].between(2, 5)]
    for col in GAP_COLUMNS:
        table = midweek.groupby(col)["ratio"].agg(["median", "size"]).loc[1:5]
        print(f"{col}（火〜金）:\n{table.round(3).to_string()}")


def censoring(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("次の休業日・祝日が分からない行の割合（予定は test の末尾までしか無い）")
    calendar = Calendar.build(train, test)
    gaps = calendar.gaps(Gaps(), calendar.end)
    tail_start = train["Date"].max() - pd.Timedelta(days=47)
    periods = {
        "test": gaps["Date"] >= test["Date"].min(),
        "train 末尾 48 日": gaps["Date"].between(tail_start, train["Date"].max()),
        "それより前の train": gaps["Date"] < tail_start,
    }
    table = pd.DataFrame(
        {name: gaps.loc[rows, GAP_COLUMNS].isna().mean() for name, rows in periods.items()}
    )
    print(table.round(4).to_string())
    capped = pd.DataFrame(
        {name: (gaps.loc[rows, GAP_COLUMNS] == Gaps().cap).mean() for name, rows in periods.items()}
    )
    print(f"上限（{Gaps().cap} 日）に張り付いた行の割合:\n{capped.round(4).to_string()}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    train, test, store = load_train(), load_test(), load_store()
    overview(train, test, store)
    open_and_zero_sales(train, test)
    target_distribution(train)
    calendar_effects(train)
    store_coverage(train, test)
    store_attributes(train, store)
    train_test_shift(train, test)
    time_structure(train)
    closure_effect(train, test)
    censoring(train, test)
    print(f"\n図: {FIG_DIR}")


if __name__ == "__main__":
    main()
