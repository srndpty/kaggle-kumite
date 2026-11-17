"""P2 検証分割。主評価は時系列分割。H2 の比較用に、同じ検証行をランダムに分ける分割も持つ。

実行すると分割の診断を表示する:
uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/cv.py
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from dataset import DATE, STORE, load_test, load_visits

from kumite.crossfit import Split, split_roles
from kumite.timesplit import expanding_window_splits, window_valid_rows

SEED = 42
HORIZON_DAYS = 39  # test の予測期間の長さ（P0 で実測）
N_WINDOWS = 3

SplitMaker = Callable[[pd.DataFrame, np.ndarray], list[Split]]


def time_splits(train: pd.DataFrame, test_stores: np.ndarray) -> list[Split]:
    """train の末尾から 39 日の窓を 3 つとり、各窓を窓の開始日より前の全行で学習する。"""
    return expanding_window_splits(train[DATE], train[STORE], test_stores, HORIZON_DAYS, N_WINDOWS)


def random_splits(train: pd.DataFrame, test_stores: np.ndarray) -> list[Split]:
    """時系列分割と同じ検証行を、ランダムに N_WINDOWS 個に分ける。学習は自分の検証行以外の全行。

    評価する行は時系列分割と同じだが、学習の条件は「検証行より未来の行を学習できる」「学習行が多い」
    「時間的に近い行から補間できる」がまとめて違う。差はこれらを合わせた楽観の大きさ。
    """
    valid_all = np.concatenate(
        window_valid_rows(
            train[DATE], train[STORE], test_stores, train[DATE].max(), HORIZON_DAYS, N_WINDOWS
        )
    )
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


def main() -> None:
    train, test = load_visits(), load_test()
    test_stores = test[STORE].unique()
    print(f"学習・検証に使う行: {len(train)}。test の店舗: {len(test_stores)}")
    for name, (version, make) in SPLITS.items():
        splits = make(train, test_stores)
        roles = split_roles(len(train), splits)
        print(
            f"\n[{name}] split_version: {version}（どの分割でも検証されない行 {(roles != 2).all(axis=0).sum()}）"
        )
        for k, (tr, va) in enumerate(splits):
            d_tr, d_va = train[DATE].iloc[tr], train[DATE].iloc[va]
            print(
                f"  分割 {k}: 学習 {len(tr)} 行（{d_tr.min().date()} 〜 {d_tr.max().date()}）、"
                f"検証 {len(va)} 行（{d_va.min().date()} 〜 {d_va.max().date()}、"
                f"店舗 {train[STORE].iloc[va].nunique()}）"
            )


if __name__ == "__main__":
    main()
