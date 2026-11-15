"""特徴量。ラベルを使わないもの（行の属性・カレンダー）と、ラベルを使うもの（過去売上の統計）を分ける。

ラベルを使う特徴は `SupervisedFeature` の形にし、分割ごとに学習行とその売上だけを受け取る。
"""

import itertools
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from cv import inner_blocks

from kumite.crossfit import folds_to_splits

CATEGORICAL = ["Store", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
_CODES = {
    "StateHoliday": ["0", "a", "b", "c"],
    "StoreType": ["a", "b", "c", "d"],
    "Assortment": ["a", "b", "c"],
    "PromoInterval": ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"],
}

# 生の列 + 日付の分解 + store.csv の結合
BASE_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "DayOfYear",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]
GAP_COLUMNS = ["DaysSinceClosed", "DaysUntilClosed", "DaysSinceHoliday", "DaysUntilHoliday"]


@dataclass(frozen=True)
class Gaps:
    """H4: 前後の休業日・祝日までの日数（カレンダー由来。予測期間の Open・祝日の予定も使う）。

    予定が分かるのは予測期間の末尾まで。それより先のイベントは見えないので、末尾に近い行の
    「次のイベントまでの日数」は分からない（打ち切り）。

    - censor: 検証でも、検証窓の末尾より先の予定を見ない（test と同じ打ち切りを再現する）。
      False だと、検証窓の行は窓より先の予定まで見えてしまう
    - fill_unknown: 分からない日数を cap で埋める。False なら NaN のまま
    """

    cap: int = 30
    censor: bool = False
    fill_unknown: bool = False

    @property
    def name(self) -> str:
        parts = [f"gaps{self.cap}"]
        parts += ["censored"] if self.censor else []
        parts += ["filled"] if self.fill_unknown else []
        return "-".join(parts)


def _days_to_events(days: np.ndarray, is_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """日付（通算日）の昇順に並んだ 1 店舗について、(直前のイベントからの日数, 直後のイベントまでの日数)。

    当日がイベントなら 0。該当するイベントが無ければ NaN。
    """
    events = days[is_event]
    since = np.full(len(days), np.nan)
    until = np.full(len(days), np.nan)
    if len(events) == 0:
        return since, until
    prev_pos = np.searchsorted(events, days, side="right") - 1
    next_pos = np.searchsorted(events, days, side="left")
    has_prev, has_next = prev_pos >= 0, next_pos < len(events)
    since[has_prev] = days[has_prev] - events[prev_pos[has_prev]]
    until[has_next] = events[next_pos[has_next]] - days[has_next]
    return since, until


@dataclass(frozen=True)
class Calendar:
    """店舗ごとの暦。train の全行（休業日を含む）と test をつないだもの。売上は持たない。"""

    timeline: pd.DataFrame  # Store・Date の昇順。列は Store, Date, Closed, Holiday

    @classmethod
    def build(cls, train_all: pd.DataFrame, test: pd.DataFrame) -> "Calendar":
        """test の Open の欠損は営業として扱う。"""
        columns = ["Store", "Date", "Open", "StateHoliday"]
        rows = pd.concat([train_all[columns], test[columns]], ignore_index=True)
        rows = rows.sort_values(["Store", "Date"], kind="stable").reset_index(drop=True)
        return cls(
            pd.DataFrame(
                {
                    "Store": rows["Store"],
                    "Date": rows["Date"],
                    "Closed": rows["Open"] == 0,
                    "Holiday": rows["StateHoliday"] != "0",
                }
            )
        )

    @property
    def end(self) -> pd.Timestamp:
        return self.timeline["Date"].max()

    def gaps(self, spec: Gaps, known_until: pd.Timestamp) -> pd.DataFrame:
        """(Store, Date) ごとの GAP_COLUMNS。known_until より先のイベントは無いものとして数える。"""
        known = (self.timeline["Date"] <= known_until).to_numpy()
        days = self.timeline["Date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        out = np.full((len(self.timeline), len(GAP_COLUMNS)), np.nan)
        bounds = np.flatnonzero(np.diff(self.timeline["Store"].to_numpy(), prepend=-1, append=-1))
        for j, column in enumerate(("Closed", "Holiday")):
            is_event = self.timeline[column].to_numpy() & known
            for lo, hi in itertools.pairwise(bounds):
                out[lo:hi, 2 * j], out[lo:hi, 2 * j + 1] = _days_to_events(
                    days[lo:hi], is_event[lo:hi]
                )
        out = np.minimum(out, spec.cap)
        if spec.fill_unknown:
            out = np.nan_to_num(out, nan=float(spec.cap))
        gaps = pd.DataFrame(out, columns=GAP_COLUMNS)
        return pd.concat([self.timeline[["Store", "Date"]], gaps], axis=1)

    def gaps_for(self, rows: pd.DataFrame, spec: Gaps, known_until: pd.Timestamp) -> np.ndarray:
        """rows（Store, Date を持つ）の行順に並べた GAP_COLUMNS の値。"""
        gaps = self.gaps(spec, known_until)
        merged = rows[["Store", "Date"]].merge(
            gaps, on=["Store", "Date"], how="left", validate="one_to_one"
        )
        return merged[GAP_COLUMNS].to_numpy()


def row_features(rows: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """行の属性だけで決まる特徴量（BASE_COLUMNS）。行順は rows のまま。"""
    x = rows[["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]].merge(
        store, on="Store", how="left", validate="many_to_one"
    )
    date = x["Date"].dt
    x = x.assign(
        Year=date.year,
        Month=date.month,
        Day=date.day,
        WeekOfYear=date.isocalendar().week.astype("int64"),
        DayOfYear=date.day_of_year,
    )
    for column, values in _CODES.items():
        x[column] = pd.Categorical(x[column], categories=values).codes  # 欠損は -1
    return x[BASE_COLUMNS].set_index(rows.index)


class SupervisedFeature(Protocol):
    """ラベルを使う特徴。学習行とその目的変数（log1p 売上）だけから、学習行と残りの行の特徴を作る。"""

    @property
    def name(self) -> str: ...

    @property
    def columns(self) -> list[str]: ...

    def __call__(
        self, train: pd.DataFrame, log_sales: np.ndarray, rest: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]: ...


_STAT_KEYS: dict[str, list[str]] = {
    "StoreMean": ["Store"],
    "StoreDowMean": ["Store", "DayOfWeek"],
    "StorePromoMean": ["Store", "Promo"],
    "StoreDowPromoMean": ["Store", "DayOfWeek", "Promo"],
}


def _group_means(ref: pd.DataFrame, log_sales: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    """ref の行の log 売上の平均を、キーごとに target の行へ引く。ref に無いキーは NaN。"""
    ref = ref.assign(_y=log_sales)
    out = np.full((len(target), len(_STAT_KEYS)), np.nan)
    for j, keys in enumerate(_STAT_KEYS.values()):
        means = ref.groupby(keys)["_y"].mean().rename("_m").reset_index()
        out[:, j] = target[keys].merge(means, on=keys, how="left")["_m"].to_numpy()
    return out


def _month_index(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


@dataclass(frozen=True)
class RecentStats:
    """P6（1 位解法）: 店舗（× 販促）ごとの、直近の log 売上の平均。

    月 m の行には、m - lag_months 以前の windows か月ぶんの売上だけを使う。lag_months = 2 なら
    予測期間（48 日）ぶん離れるので、検証・テスト行でも学習行と同じ条件で作れる
    （8 月の行は 6 月まで、9 月の行は 7 月までの売上を使う）。使う売上は学習行のものだけ。
    """

    windows: tuple[int, ...] = (3, 12)
    lag_months: int = 2

    @property
    def name(self) -> str:
        return f"recent_stats-lag{self.lag_months}-w{'_'.join(map(str, self.windows))}"

    @property
    def columns(self) -> list[str]:
        return [f"Recent{w}m{key}Mean" for w in self.windows for key in ("Store", "StorePromo")]

    def __call__(
        self, train: pd.DataFrame, log_sales: np.ndarray, rest: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        months = _month_index(train["Date"])
        first = int(months.min())
        everyone = pd.concat([train[["Store", "Date"]], rest[["Store", "Date"]]])
        n_months = int(_month_index(everyone["Date"]).max()) - first + 1
        n_stores = int(everyone["Store"].max()) + 1
        # 月 × 店舗 × 販促の合計と件数を、月方向に累積する（先頭に 0 の行を置く）
        sums = np.zeros((n_months + 1, n_stores, 2))
        counts = np.zeros((n_months + 1, n_stores, 2))
        index = (months - first + 1, train["Store"].to_numpy(), train["Promo"].to_numpy())
        np.add.at(sums, index, log_sales)
        np.add.at(counts, index, 1)
        sums, counts = sums.cumsum(axis=0), counts.cumsum(axis=0)

        def features(rows: pd.DataFrame) -> np.ndarray:
            # 累積の行 e は「月 e - 1（first 起点）までの合計」。使える最後の月は m - lag_months
            end = np.clip(_month_index(rows["Date"]) - first + 1 - self.lag_months, 0, n_months)
            store, promo = rows["Store"].to_numpy(), rows["Promo"].to_numpy()
            out = np.full((len(rows), len(self.columns)), np.nan)
            for j, window in enumerate(self.windows):
                start = np.clip(end - window, 0, n_months)
                s = sums[end, store] - sums[start, store]  # (行, 販促)
                c = counts[end, store] - counts[start, store]
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[:, 2 * j] = s.sum(axis=1) / c.sum(axis=1)
                    rows_idx = np.arange(len(rows))
                    out[:, 2 * j + 1] = s[rows_idx, promo] / c[rows_idx, promo]
            return np.where(np.isfinite(out), out, np.nan)

        return features(train), features(rest)


@dataclass(frozen=True)
class StoreStats:
    """H3: 店舗（× 曜日 × 販促）ごとの、学習期間の log 売上の平均。

    検証・テスト行には、学習行全体から作った値を渡す。学習行は out_of_block=True なら、
    学習行を日付順に分けた期間ごとに「自分の期間以外」から作る（自分の売上が特徴に入らない）。
    False なら学習行全体から作る（自分の売上が入る。切り分け用）。
    """

    out_of_block: bool = True

    @property
    def name(self) -> str:
        return "store_stats" if self.out_of_block else "store_stats_insample"

    @property
    def columns(self) -> list[str]:
        return list(_STAT_KEYS)

    def __call__(
        self, train: pd.DataFrame, log_sales: np.ndarray, rest: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        f_rest = _group_means(train, log_sales, rest)
        if not self.out_of_block:
            return _group_means(train, log_sales, train), f_rest
        f_train = np.full((len(train), len(_STAT_KEYS)), np.nan)
        for ref, block in folds_to_splits(inner_blocks(train["Date"])):
            f_train[block] = _group_means(train.iloc[ref], log_sales[ref], train.iloc[block])
        return f_train, f_rest
