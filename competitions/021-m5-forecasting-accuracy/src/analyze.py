"""事前仮説ごとに、指定した CV run と E01 の誤差を同じ行で比較する。"""

import argparse
import json

import numpy as np
from dataset import HORIZON, ORIGINS, VALIDATION_FILE, Sales, load_sales
from train import STORE, input_hash


def segment(data: Sales, origin: int, experiment: str) -> np.ndarray:
    """仮説で効果を予想した商品×店舗×日の行を返す。"""
    days = slice(origin, origin + HORIZON)
    history = data.actual(origin - HORIZON + 1, origin)
    if experiment == "E02":
        event = data.calendar.iloc[days][["event_name_1", "event_name_2"]].notna().any(axis=1)
        snap = data.calendar.iloc[days][["snap_CA", "snap_TX", "snap_WI"]].to_numpy()
        states = data.series["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy(dtype=int)
        return (snap[:, states].T > 0) | event.to_numpy()
    if experiment == "E03":
        past = data.price(origin, origin)
        planned = data.price(origin + 1, origin + HORIZON)
        return np.isfinite(past) & np.isfinite(planned) & (past != planned)
    if experiment == "E04":
        intermittent = ((history > 0).sum(axis=1) <= 7)[:, None]
        return np.broadcast_to(intermittent, history.shape)
    raise ValueError(f"仮説別の集計が未定義です: {experiment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    baseline = STORE.record(args.baseline)
    if baseline.params.get("experiment") != "E01" or baseline.data_hash != input_hash(
        VALIDATION_FILE
    ):
        raise ValueError("E01 の記録または入力ファイルが一致しません")
    data = load_sales()
    actual = np.stack([data.actual(origin + 1, origin + HORIZON) for origin in ORIGINS])
    base_error = np.abs(actual - STORE.load_array(baseline.run_id, "predictions.npy"))
    result = {}
    for run_id in args.run_ids:
        other = STORE.record(run_id)
        if (other.data_hash, other.split_hash) != (baseline.data_hash, baseline.split_hash):
            raise ValueError(f"実験のデータまたは分割が一致しません: {run_id}")
        mask = np.stack([segment(data, origin, other.params["experiment"]) for origin in ORIGINS])
        error = np.abs(actual - STORE.load_array(run_id, "predictions.npy"))
        result[run_id] = {
            name: {
                "rows": int(group.sum()),
                "baseline_mae": float(base_error[group].mean()),
                "experiment_mae": float(error[group].mean()),
            }
            for name, group in (("hypothesis_segment", mask), ("other", ~mask))
            if group.any()
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
