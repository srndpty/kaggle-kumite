"""P6（1 位解法）: 予測に一定の係数を掛ける後処理。

log1p(売上) を RMSE で学習した予測は、売上の幾何平均あたりを狙う。RMSPE は過大予測を過小予測より
重く罰するので、予測を少し小さくした方が良い。係数は検証窓ごとに、他の 2 つの窓の OOF で決める
（その窓の売上は使わない）。テスト予測には、3 窓すべての OOF で決めた係数を掛ける。

注意（K002）: 係数は「検証窓で early stopping したモデル」の OOF で決めるが、提出は
「全行で学習し直したモデル」の予測で、形が違う。

実行: uv run python competitions/019-rossmann-store-sales/src/scale.py E10 E11
"""

import argparse
import time
from dataclasses import replace

import numpy as np
from pipeline import STORE, load_data, load_predictions, rmspe, save_run
from train import EXPERIMENTS

from kumite.crossfit import CrossFitResult

FACTORS = np.round(np.arange(0.9, 1.1 + 1e-9, 0.0005), 4)


def fit_factor(y: np.ndarray, pred: np.ndarray) -> float:
    """RMSPE を最小にする係数（0.9〜1.1 を 0.0005 刻みで探す）。"""
    return float(min(FACTORS, key=lambda f: rmspe(y, f * pred)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run_id", choices=list(EXPERIMENTS))
    parser.add_argument("run_id")
    args = parser.parse_args()
    base_run_id: str = args.base_run_id
    base = EXPERIMENTS[base_run_id]

    with STORE.reserve(args.run_id) as reservation:
        data = load_data()
        y, splits = data.y, data.splits(base.split)
        oof, test_pred = load_predictions(base_run_id)
        started = time.perf_counter()
        scaled = oof.copy()
        infos = []
        for k, (_, valid) in enumerate(splits):
            others = np.concatenate([va for j, (_, va) in enumerate(splits) if j != k])
            factor = fit_factor(y[others], oof[others])
            scaled[valid] = factor * oof[valid]
            infos.append({"factor": round(factor, 5)})
            print(
                f"分割 {k}: 係数 {factor:.5f}（この窓にとって最適な係数 {fit_factor(y[valid], oof[valid]):.5f}）"
            )
        validated = np.concatenate([va for _, va in splits])
        factor_all = fit_factor(y[validated], oof[validated])
        print(f"テスト予測に掛ける係数: {factor_all:.5f}")

        exp = replace(
            base,
            model=f"{base_run_id}+scale",
            hypothesis="H10（1 位解法）: log 学習の予測に 1 未満の係数を掛けると RMSPE が下がる",
            changed_one_thing=f"{base_run_id} の予測に係数を掛ける（係数は他の検証窓の OOF で決める）",
            params={"base_run_id": base_run_id, "factor_for_test": round(factor_all, 5)},
        )
        result = CrossFitResult(scaled, factor_all * test_pred, infos)
        inputs = STORE.input_hashes(base_run_id, "oof.npy", "test.npy")
        save_run(reservation, exp, data, splits, result, time.perf_counter() - started, inputs)


if __name__ == "__main__":
    main()
