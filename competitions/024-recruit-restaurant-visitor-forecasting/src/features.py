"""特徴量。行の属性だけで決まるもの（`row_features`）と、分割ごとに作るもの（`Feature`）に分ける。

分割ごとに作る特徴は `FoldContext` を受け取り、(学習行, 検証行, テスト行) の値を返す。
来店数（ラベル）は学習行の分だけが `FoldContext.log_y` で渡される。
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from cv import HORIZON_DAYS, SEED
from dataset import DATE, STORE

CATEGORICAL = ["store", "genre", "pref", "city"]
DAY = pd.Timedelta(days=1)


def row_features(rows: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """店舗の属性と日付の分解。カテゴリは店舗情報の並びで番号を振る（学習・テストで共通）。"""
    area = stores["air_area_name"].str.split(" ")
    info = pd.DataFrame(
        {
            STORE: stores[STORE],
            "store": np.arange(len(stores)),
            "genre": pd.factorize(stores["air_genre_name"])[0],
            "pref": pd.factorize(area.str[0])[0],
            "city": pd.factorize(area.str[0] + " " + area.str[1])[0],
            "lat": stores["latitude"],
            "lon": stores["longitude"],
        }
    )
    out = rows[[STORE]].merge(info, on=STORE, how="left").drop(columns=STORE)
    if out["store"].isna().any():
        raise ValueError("店舗情報の無い店舗があります")
    date = rows[DATE].reset_index(drop=True)
    return out.assign(
        dow=date.dt.day_of_week,
        year=date.dt.year,
        month=date.dt.month,
        day=date.dt.day,
        weekofyear=date.dt.isocalendar().week.astype(np.int64),
        dayofyear=date.dt.day_of_year,
    ).set_index(rows.index)


@dataclass(frozen=True)
class FoldContext:
    """1 分割で特徴を作るのに使ってよい情報。"""

    train: pd.DataFrame  # 学習行
    valid: pd.DataFrame  # 検証行（全行で学習し直すときは空）
    test: pd.DataFrame
    log_y: np.ndarray  # 学習行の log1p(来店数)。検証行・テスト行のラベルは渡さない
    reserves: pd.DataFrame  # 予約（`dataset.load_reserves`）
    holidays: pd.Series  # 日付 → 祝日フラグ（`date_info` の全期間）
    valid_origin: pd.Timestamp  # 検証窓の予測の起点（窓の開始日）
    test_origin: pd.Timestamp
    fit_known_end: pd.Timestamp  # 学習・検証行に見せる予定の末尾（検証窓の末尾）
    test_known_end: pd.Timestamp  # テスト行に見せる予定の末尾（test の末尾）


Built = tuple[np.ndarray, np.ndarray, np.ndarray]


class Feature(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def columns(self) -> list[str]: ...

    def build(self, ctx: FoldContext) -> Built: ...


# --- 祝日（ラベルを使わない。予定は known_end までしか見せない） ---


@dataclass(frozen=True)
class Holidays:
    """その日が祝日か、前日・翌日が休日（土日祝）か。known_end より先の祝日は分からないものとして 0。"""

    @property
    def name(self) -> str:
        return "holidays-censored"

    @property
    def columns(self) -> list[str]:
        return ["Holiday", "PrevOff", "NextOff"]

    def build(self, ctx: FoldContext) -> Built:
        def make(rows: pd.DataFrame, known_end: pd.Timestamp) -> np.ndarray:
            def holiday(dates: pd.Series) -> np.ndarray:
                flag = dates.map(ctx.holidays).fillna(0).to_numpy(dtype=np.float64)
                return np.where(dates <= known_end, flag, 0.0)

            def off(dates: pd.Series) -> np.ndarray:
                return np.maximum(holiday(dates), (dates.dt.day_of_week >= 5).to_numpy())

            dates = rows[DATE]
            return np.column_stack([holiday(dates), off(dates - DAY), off(dates + DAY)])

        return (
            make(ctx.train, ctx.fit_known_end),
            make(ctx.valid, ctx.fit_known_end),
            make(ctx.test, ctx.test_known_end),
        )


# --- 店舗別の過去の来店数（ラベルを使う） ---


def _past_mean(
    ref: pd.DataFrame, values: np.ndarray, query: pd.DataFrame, keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """query の各行について、ref のうち同じキーで日付がその日より前の行の values の (平均, 行数)。"""
    agg = (
        ref[[*keys, DATE]]
        .assign(_v=values)
        .groupby([*keys, DATE])["_v"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(DATE, kind="stable")
    )
    agg[["csum", "ccount"]] = agg.groupby(keys)[["sum", "count"]].cumsum()
    q = query[[*keys, DATE]].reset_index(drop=True).reset_index(names="_row")
    merged = pd.merge_asof(
        q.sort_values(DATE, kind="stable"),
        agg[[*keys, DATE, "csum", "ccount"]],
        on=DATE,
        by=keys,
        allow_exact_matches=False,
    ).sort_values("_row")
    count = merged["ccount"].fillna(0).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = merged["csum"].to_numpy(dtype=np.float64) / count
    return mean, count


@dataclass(frozen=True)
class StoreStats:
    """店舗・店舗 × 曜日の log1p 来店数の平均と行数。どの行も、学習行のうちその日より前の行だけから作る。

    時系列分割の検証・テスト行では「学習行すべて」と同じ（学習行はすべて窓の開始より前）。
    """

    @property
    def name(self) -> str:
        return "store_stats-past"

    @property
    def columns(self) -> list[str]:
        return ["StoreMean", "StoreCount", "StoreDowMean", "StoreDowCount"]

    def build(self, ctx: FoldContext) -> Built:
        ref = ctx.train.assign(dow=ctx.train[DATE].dt.day_of_week)

        def make(rows: pd.DataFrame) -> np.ndarray:
            query = rows.assign(dow=rows[DATE].dt.day_of_week)
            store = _past_mean(ref, ctx.log_y, query, [STORE])
            store_dow = _past_mean(ref, ctx.log_y, query, [STORE, "dow"])
            return np.column_stack([*store, *store_dow])

        return make(ctx.train), make(ctx.valid), make(ctx.test)


# --- 予約（ラベルを使わない。予約日時で打ち切る） ---


def _reserved(reserves: pd.DataFrame, rows: pd.DataFrame, cutoff: pd.Series) -> np.ndarray:
    """各行の店舗・日付への予約のうち、予約日時が cutoff より前のものの (人数, 件数)。"""
    r = reserves.assign(**{DATE: reserves["visit_datetime"].dt.normalize()})
    q = rows[[STORE, DATE]].reset_index(drop=True).assign(_cutoff=cutoff.to_numpy())
    q = q.reset_index(names="_row")
    joined = q.merge(r[[STORE, DATE, "reserve_datetime", "reserve_visitors"]], on=[STORE, DATE])
    joined = joined[joined["reserve_datetime"] < joined["_cutoff"]]
    agg = joined.groupby("_row")["reserve_visitors"].agg(["sum", "count"])
    agg = agg.reindex(range(len(rows)), fill_value=0)
    return agg.to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class Reservations:
    """予約人数・件数と、予測の起点からの日数（何日先の予測か）。

    検証・テスト行は予測の起点（窓の開始日の 0 時）より前に入った予約だけを数える（test と同じ）。

    - train_rows="horizon": 学習行にも、0〜38 日の「何日先か」を行ごとにランダムに割り当て、その日数ぶん
      前の 0 時より前に入った予約だけを数える（検証・テスト行と同じ打ち切りの分布にする）
    - train_rows="all": 学習行は来店日までに入った予約をすべて数え、何日先かは 0 とする
    - valid_rows="all": 検証行でも打ち切らない（H3b。test では再現しない楽観を測るためだけに使う）
    """

    train_rows: Literal["horizon", "all"] = "horizon"
    valid_rows: Literal["cutoff", "all"] = "cutoff"

    @property
    def name(self) -> str:
        return f"reservations-train_{self.train_rows}-valid_{self.valid_rows}"

    @property
    def columns(self) -> list[str]:
        return ["ReserveVisitors", "ReserveCount", "DaysAhead"]

    def build(self, ctx: FoldContext) -> Built:
        def make(rows: pd.DataFrame, cutoff: pd.Series, ahead: np.ndarray) -> np.ndarray:
            return np.column_stack([_reserved(ctx.reserves, rows, cutoff), ahead])

        def from_origin(rows: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.Series, np.ndarray]:
            ahead = (rows[DATE] - origin).dt.days.to_numpy(dtype=np.float64)
            if (ahead < 0).any():
                raise ValueError(f"予測の起点 {origin.date()} より前の行があります")
            return pd.Series(origin, index=rows.index), ahead

        dates = ctx.train[DATE]
        if self.train_rows == "horizon":
            ahead_tr = np.random.default_rng(SEED).integers(0, HORIZON_DAYS, size=len(dates))
            cut_tr = dates - pd.to_timedelta(ahead_tr, unit="D")
        else:
            ahead_tr = np.zeros(len(dates), dtype=np.int64)
            cut_tr = dates + DAY
        cut_va, ahead_va = from_origin(ctx.valid, ctx.valid_origin)
        if self.valid_rows == "all":
            cut_va = ctx.valid[DATE] + DAY
        cut_te, ahead_te = from_origin(ctx.test, ctx.test_origin)
        return (
            make(ctx.train, cut_tr, ahead_tr.astype(np.float64)),
            make(ctx.valid, cut_va, ahead_va),
            make(ctx.test, cut_te, ahead_te),
        )
