"""P2 検証分割。主評価は時系列分割。H2 の比較用に、同じ検証行をランダムに分ける分割も持つ。

実行すると分割の診断を表示する:
uv run python competitions/019-rossmann-store-sales/src/cv.py
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from dataset import load_test, load_train, modeling_rows

from kumite.crossfit import Split, split_roles
from kumite.timesplit import expanding_window_splits, window_valid_rows

SEED = 42
HORIZON_DAYS = 48  # test の予測期間の長さ（P1 で実測）
N_WINDOWS = 3
N_INNER_BLOCKS = 4

SplitMaker = Callable[[pd.DataFrame, np.ndarray], list[Split]]


def _valid_rows(
    train: pd.DataFrame, test_stores: np.ndarray, last_day: pd.Timestamp
) -> list[np.ndarray]:
    """last_day で終わる窓ごとの検証行。test に出る店舗だけを検証する。"""
    return window_valid_rows(
        train["Date"], train["Store"], test_stores, last_day, HORIZON_DAYS, N_WINDOWS
    )


def time_splits(
    train: pd.DataFrame, test_stores: np.ndarray, last_day: pd.Timestamp | None = None
) -> list[Split]:
    """各窓を、窓の開始日より前の全行で学習して検証する（expanding window）。

    last_day を外側の検証窓の前日にすると、外側の学習期間の中だけで閉じた内側の分割になる。
    """
    return expanding_window_splits(
        train["Date"], train["Store"], test_stores, HORIZON_DAYS, N_WINDOWS, last_day
    )


def random_splits(train: pd.DataFrame, test_stores: np.ndarray) -> list[Split]:
    """時系列分割と同じ検証行を、ランダムに N_WINDOWS 個に分ける。学習は自分の検証行以外の全行。

    検証行の前後の日・同じ日の他店舗・未来の行が学習側に入る。評価する行は時系列分割と同じだが、
    学習の条件は次の点がまとめて違う。スコアの差はこれらを合わせた楽観の大きさで、内訳は分けていない。

    - 検証行より未来の行を学習できる
    - 学習行が多い（約 81 万行。時系列分割は 71〜80 万行）
    - 検証行と時間的に近い行（前後の日・同じ日の他店舗）から補間できる
    """
    valid_all = np.concatenate(_valid_rows(train, test_stores, train["Date"].max()))
    assignment = np.random.default_rng(SEED).integers(0, N_WINDOWS, size=len(valid_all))
    all_rows = np.arange(len(train))
    splits: list[Split] = []
    for k in range(N_WINDOWS):
        valid = np.sort(valid_all[assignment == k])
        splits.append((np.setdiff1d(all_rows, valid, assume_unique=True), valid))
    return splits


# 名前 → (split_version, 分割の作り方)
SPLITS: dict[str, tuple[str, SplitMaker]] = {
    "time": (f"ts{N_WINDOWS}-{HORIZON_DAYS}d-teststores-v1", time_splits),
    "random": (f"random{N_WINDOWS}-seed{SEED}-same-valid-rows-v1", random_splits),
}


def inner_blocks(dates: pd.Series) -> np.ndarray:
    """学習行を日付順に N_INNER_BLOCKS 個の連続した期間に分けた fold 番号。

    ラベルを使う特徴量を、学習行について「自分の期間の売上を使わない値」にするために使う。
    """
    days = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    edges = np.quantile(days, np.linspace(0, 1, N_INNER_BLOCKS + 1)[1:-1])
    return np.searchsorted(edges, days, side="right").astype(np.int8)


def main() -> None:
    train, test = modeling_rows(load_train()), load_test()
    test_stores = test["Store"].unique()
    print(f"学習・検証に使う行（売上 > 0）: {len(train)}。test の店舗: {len(test_stores)}")
    for name, (version, make) in SPLITS.items():
        splits = make(train, test_stores)
        roles = split_roles(len(train), splits)
        print(
            f"\n[{name}] split_version: {version}（どの分割でも検証されない行 {(roles != 2).all(axis=0).sum()}）"
        )
        for k, (tr, va) in enumerate(splits):
            d_tr, d_va = train["Date"].iloc[tr], train["Date"].iloc[va]
            print(
                f"  分割 {k}: 学習 {len(tr)} 行（{d_tr.min().date()} 〜 {d_tr.max().date()}）、"
                f"検証 {len(va)} 行（{d_va.min().date()} 〜 {d_va.max().date()}、"
                f"店舗 {train['Store'].iloc[va].nunique()}）"
            )
    time_valid = np.sort(np.concatenate([va for _, va in SPLITS["time"][1](train, test_stores)]))
    random_valid = np.sort(
        np.concatenate([va for _, va in SPLITS["random"][1](train, test_stores)])
    )
    print(f"\n2 つの分割で検証行が同じ: {np.array_equal(time_valid, random_valid)}")
    blocks = inner_blocks(train["Date"])
    for b in range(N_INNER_BLOCKS):
        d = train["Date"][blocks == b]
        print(
            f"内側の期間 {b}（全行で作った例）: {d.min().date()} 〜 {d.max().date()}、{len(d)} 行"
        )


if __name__ == "__main__":
    main()
