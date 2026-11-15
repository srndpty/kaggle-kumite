"""P7 エラー分析と、小さい差の確認（店舗単位の bootstrap）。

実行:
  uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/analyze.py errors E05
  uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/analyze.py compare E02 E03 E04
"""

import argparse

import numpy as np
import pandas as pd
from cv import HORIZON_DAYS, N_WINDOWS, SEED
from dataset import DATA_FILES, DATE, STORE, TARGET, load_date_info, load_visits
from pipeline import STORE_RUNS

from kumite.bootstrap import group_bootstrap_diff
from kumite.experiments import compare_runs
from kumite.provenance import files_sha256
from kumite.timesplit import windows

N_BOOTSTRAP = 2000


def _validated(run_id: str) -> pd.DataFrame:
    """検証された行に、予測と log 誤差（log1p(予測) - log1p(実績)）を付けて返す。"""
    if STORE_RUNS.record(run_id).data_hash != files_sha256(DATA_FILES):
        raise ValueError(f"{run_id} の記録の data_hash が、今のデータファイルと一致しません")
    train = load_visits()
    pred = STORE_RUNS.load_array(run_id, "oof.npy")
    ok = np.isfinite(pred)
    rows = train[ok].assign(pred=pred[ok])
    return rows.assign(err=np.log1p(rows["pred"]) - np.log1p(rows[TARGET]))


def _rmsle(err: pd.Series) -> float:
    return float(np.sqrt((err**2).mean()))


def _by(rows: pd.DataFrame, key: pd.Series | str, label: str) -> None:
    table = rows.groupby(key)["err"].agg(
        rmsle=_rmsle, bias="median", rows="size", share=lambda e: (e**2).sum()
    )
    table["share"] = table["share"] / (rows["err"] ** 2).sum()
    print(f"\n{label}:\n{table.round(4).to_string()}")


def errors(run_id: str) -> None:
    rows = _validated(run_id)
    print(
        f"{run_id}: RMSLE {_rmsle(rows['err']):.5f}、log 誤差の中央値 {rows['err'].median():+.4f}"
    )
    starts = [s for s, _ in windows(rows[DATE].max(), HORIZON_DAYS, N_WINDOWS)]
    window = (
        np.searchsorted(np.array(starts, dtype="datetime64[ns]"), rows[DATE].to_numpy(), "right")
        - 1
    )
    rows = rows.assign(window=window)
    ahead = (rows[DATE] - pd.to_datetime(np.array(starts)[window])).dt.days
    holidays = load_date_info().set_index("calendar_date")["holiday_flg"]
    _by(rows, "window", "検証窓ごと（bias は log 誤差の中央値、share は二乗誤差に占める割合）")
    _by(rows, (ahead // 7).rename("窓の開始からの週"), "予測の遠さごと")
    _by(rows, rows[DATE].dt.day_of_week.rename("曜日（月=0）"), "曜日ごと")
    _by(rows, rows[DATE].map(holidays).rename("祝日"), "祝日かどうか")
    squared = (rows["err"] ** 2).sort_values(ascending=False)
    for share in (0.01, 0.05):
        top = squared.head(int(len(squared) * share)).sum() / squared.sum()
        print(f"log 誤差の大きい上位 {share:.0%} の行が二乗誤差に占める割合: {top:.3f}")
    worst = rows.reindex(squared.index[:8])[[STORE, DATE, TARGET, "pred", "err"]]
    print(f"\n誤差の大きい行:\n{worst.round(3).to_string()}")


def _same_rows(base_run_id: str, base: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """run_id の検証行を返す。評価条件と検証行の (店舗, 日付) の並びが基準と一致しなければ拒否する。"""
    compare_runs(STORE_RUNS.record(base_run_id), STORE_RUNS.record(run_id))
    other = _validated(run_id)
    if not other[[STORE, DATE]].equals(base[[STORE, DATE]]):
        raise ValueError(f"{run_id} の検証行（店舗, 日付）が基準 {base_run_id} と一致しません")
    return other


def compare(base_run_id: str, run_ids: list[str]) -> None:
    """行ごとの二乗 log 誤差の差を、店舗単位で再標本化して確かめる（同じ店舗の行は独立でないため）。"""
    base = _validated(base_run_id)
    print(
        f"基準 {base_run_id}: RMSLE {_rmsle(base['err']):.5f}。差は (実験 - 基準)、{N_BOOTSTRAP} 回の店舗単位 bootstrap"
    )
    for run_id in run_ids:
        other = _same_rows(base_run_id, base, run_id)
        d = group_bootstrap_diff(
            (base["err"] ** 2).to_numpy(),
            (other["err"] ** 2).to_numpy(),
            base[STORE].to_numpy(),
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )
        print(
            f"{run_id}: 差 {d.point:+.5f}、95% 区間 [{d.low:+.5f}, {d.high:+.5f}]、"
            f"改善した再標本の割合 {d.share_negative:.3f}"
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
