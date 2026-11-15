"""P7 エラー分析と、小さい差の確認（店舗単位の bootstrap）。

実行:
  uv run python competitions/019-rossmann-store-sales/src/analyze.py errors E09
  uv run python competitions/019-rossmann-store-sales/src/analyze.py compare E09 E08 E10 E11
"""

import argparse

import numpy as np
import pandas as pd
from cv import SEED, windows
from dataset import DATA_FILES, TARGET, load_train, modeling_rows
from pipeline import STORE

from kumite.experiments import compare_runs
from kumite.provenance import files_sha256

N_BOOTSTRAP = 2000


def _validated(run_id: str) -> pd.DataFrame:
    """検証された行に、予測と相対誤差を付けて返す。run の記録と今のデータが同じことを確かめる。"""
    if STORE.record(run_id).data_hash != files_sha256(DATA_FILES):
        raise ValueError(f"{run_id} の記録の data_hash が、今のデータファイルと一致しません")
    train = modeling_rows(load_train())
    pred = STORE.load_array(run_id, "oof.npy")
    rows = train[np.isfinite(pred)].assign(pred=pred[np.isfinite(pred)])
    return rows.assign(pct=(rows["pred"] - rows[TARGET]) / rows[TARGET])


def _rmspe(pct: pd.Series) -> float:
    return float(np.sqrt((pct**2).mean()))


def _by(rows: pd.DataFrame, key: pd.Series | str, label: str) -> None:
    table = rows.groupby(key)["pct"].agg(
        rmspe=_rmspe, bias="median", rows="size", share=lambda p: (p**2).sum()
    )
    table["share"] = table["share"] / (rows["pct"] ** 2).sum()
    print(f"\n{label}:\n{table.round(4).to_string()}")


def errors(run_id: str) -> None:
    rows = _validated(run_id)
    print(
        f"{run_id}: RMSPE {_rmspe(rows['pct']):.5f}、相対誤差の中央値 {rows['pct'].median():+.4f}"
    )
    print(f"過大予測の行の割合: {(rows['pct'] > 0).mean():.4f}")
    starts = [start for start, _ in windows(rows["Date"].max())]
    window = (
        np.searchsorted(np.array(starts, dtype="datetime64[ns]"), rows["Date"].to_numpy(), "right")
        - 1
    )
    rows = rows.assign(window=window)
    day = (rows["Date"] - pd.to_datetime(np.array(starts)[window])).dt.days
    _by(rows, "window", "検証窓ごと（bias は相対誤差の中央値、share は二乗誤差に占める割合）")
    _by(rows, (day // 12).rename("窓の開始からの日数 // 12"), "予測の遠さごと")
    _by(rows, "DayOfWeek", "曜日ごと")
    _by(rows, "Promo", "販促ごと")
    _by(rows, (rows["StateHoliday"] != "0").rename("祝日"), "祝日かどうか")

    # 誤差の集中: 相対誤差の絶対値が大きい行が、二乗誤差のどれだけを占めるか
    squared = (rows["pct"] ** 2).sort_values(ascending=False)
    for share in (0.001, 0.01, 0.05):
        top = squared.head(int(len(squared) * share)).sum() / squared.sum()
        print(f"相対誤差の大きい上位 {share:.1%} の行が二乗誤差に占める割合: {top:.3f}")
    by_store = rows.groupby("Store")["pct"].agg(rmspe=_rmspe, bias="median", rows="size")
    print(f"\n店舗ごとの RMSPE の分布:\n{by_store['rmspe'].describe().round(4).to_string()}")
    worst_stores = by_store.loc[by_store["rmspe"].nlargest(8).index]
    print(f"RMSPE が大きい店舗:\n{worst_stores.round(4).to_string()}")
    worst = rows.reindex(squared.index[:8])[
        ["Store", "Date", "DayOfWeek", "Promo", TARGET, "pred", "pct"]
    ]
    print(f"\n相対誤差が最大の行:\n{worst.to_string(index=False, float_format='%.3f')}")


def _same_rows(base_run_id: str, base: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """run_id の検証行を返す。評価条件（データ・分割・指標）と、検証行の (Store, Date) と並びが
    基準と一致しなければ拒否する（店舗ごとの集計を基準の行順で行うため）。"""
    compare_runs(STORE.record(base_run_id), STORE.record(run_id))
    other = _validated(run_id)
    if not other[["Store", "Date"]].equals(base[["Store", "Date"]]):
        raise ValueError(f"{run_id} の検証行（Store, Date）が基準 {base_run_id} と一致しません")
    return other


def compare(base_run_id: str, run_ids: list[str]) -> None:
    """行ごとの二乗相対誤差の差を、店舗単位で再標本化して確かめる（同じ店舗の行は独立でないため）。"""
    base = _validated(base_run_id)
    stores = base["Store"].to_numpy()
    codes, n_stores = pd.factorize(stores)[0], len(np.unique(stores))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_stores, size=(N_BOOTSTRAP, n_stores))
    counts = np.bincount(codes, minlength=n_stores)

    def store_sums(pct: pd.Series) -> np.ndarray:
        return np.bincount(codes, weights=(pct**2).to_numpy(), minlength=n_stores)

    base_sums = store_sums(base["pct"])
    print(
        f"基準 {base_run_id}: RMSPE {_rmspe(base['pct']):.5f}。差は (実験 - 基準)、{N_BOOTSTRAP} 回の店舗単位 bootstrap"
    )
    for run_id in run_ids:
        other_sums = store_sums(_same_rows(base_run_id, base, run_id)["pct"])
        n = counts[draws].sum(axis=1)
        diff = np.sqrt(other_sums[draws].sum(axis=1) / n) - np.sqrt(
            base_sums[draws].sum(axis=1) / n
        )
        lo, hi = np.percentile(diff, [2.5, 97.5])
        point = np.sqrt(other_sums.sum() / counts.sum()) - np.sqrt(base_sums.sum() / counts.sum())
        print(
            f"{run_id}: 差 {point:+.5f}、95% 区間 [{lo:+.5f}, {hi:+.5f}]、"
            f"改善した再標本の割合 {(diff < 0).mean():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("errors").add_argument("run_id")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("base")
    compare_parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    if args.command == "errors":
        errors(args.run_id)
    else:
        compare(args.base, args.run_ids)


if __name__ == "__main__":
    main()
