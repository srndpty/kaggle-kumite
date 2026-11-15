"""P6（1 位解法）: 予測に一定の係数を掛ける後処理。

log1p(売上) を RMSE で学習した予測は、売上の幾何平均あたりを狙う。RMSPE は過大予測を過小予測より
重く罰するので、予測を少し小さくした方が良い。

係数の決め方:
- 外側の検証窓 k の係数は、その分割の学習期間（窓 k の開始日より前）の中だけで内側の時系列 CV を
  組み直し、その OOF で決める。窓 k 以降の売上も、窓 k の売上で学習したモデルの予測も使わない。
  外側の学習期間以外の売上を変えても係数が変わらないことを、実行のたびに確かめる
- テスト予測には、外側の 3 窓すべての OOF で決めた係数を掛ける

注意（K002）: 係数は「検証窓で early stopping したモデル」の OOF で決めるが、提出は
「全行で学習し直したモデル」の予測で、形が違う。

実行: uv run python competitions/019-rossmann-store-sales/src/scale.py E09 E11-r2
"""

import argparse
import time
from dataclasses import replace

import numpy as np
from cv import time_splits
from pipeline import (
    STORE,
    Data,
    Experiment,
    fit_split,
    load_data,
    load_predictions,
    rmspe,
    save_run,
)
from train import EXPERIMENTS

from kumite.checks import check_label_independence, perturb_targets
from kumite.crossfit import CrossFitResult, cross_fit_splits

FACTORS = np.round(np.arange(0.9, 1.1 + 1e-9, 0.0005), 4)


def fit_factor(y: np.ndarray, pred: np.ndarray) -> float:
    """RMSPE を最小にする係数（0.9〜1.1 を 0.0005 刻みで探す）。"""
    return float(min(FACTORS, key=lambda f: rmspe(y, f * pred)))


def inner_oof(
    exp: Experiment, data: Data, y: np.ndarray, outer_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """outer_train の期間の中だけで内側の時系列 CV をし、(内側の検証行, その OOF 予測) を返す。

    内側の窓は outer_train の最後の日で終わる。内側の学習行・検証行はすべて outer_train に含まれる。
    """
    last_day = data.train["Date"].iloc[outer_train].max()
    inner = time_splits(data.train, data.test["Store"].unique(), last_day)
    used = np.concatenate([np.concatenate([tr, va]) for tr, va in inner])
    if not np.isin(used, outer_train).all():
        raise ValueError("内側の分割が外側の学習行の外に出ています")
    result = cross_fit_splits(len(y), inner, lambda tr, va: fit_split(exp, data, y, tr, va))
    rows = np.sort(np.concatenate([va for _, va in inner]))
    return rows, result.oof[rows]


def outer_factor(exp: Experiment, data: Data, y: np.ndarray, outer_train: np.ndarray) -> float:
    """外側の 1 分割に掛ける係数。外側の学習行以外の売上を変えても係数が変わらないことを確かめる。"""
    rows, pred = inner_oof(exp, data, y, outer_train)
    factor = fit_factor(y[rows], pred)

    hidden = np.ones(len(y), dtype=bool)
    hidden[outer_train] = False

    def build(labels: np.ndarray) -> list[np.ndarray]:
        if labels is y:  # 上で計算済み
            return [pred, np.array([factor])]
        rows_p, pred_p = inner_oof(exp, data, labels, outer_train)
        assert np.array_equal(rows_p, rows)
        return [pred_p, np.array([fit_factor(labels[rows], pred_p)])]

    check_label_independence(build, y, perturb_targets(y, hidden))
    return factor


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
        for k, (outer_train, valid) in enumerate(splits):
            factor = outer_factor(base, data, y, outer_train)
            scaled[valid] = factor * oof[valid]
            infos.append({"factor": round(factor, 5)})
            print(
                f"分割 {k}: 係数 {factor:.5f}（外側の学習期間内の内側 CV で決定。検証ラベルの摂動で不変）、"
                f"参考: この窓にとって最適な係数 {fit_factor(y[valid], oof[valid]):.5f}",
                flush=True,
            )
        validated = np.concatenate([va for _, va in splits])
        factor_all = fit_factor(y[validated], oof[validated])
        print(f"テスト予測に掛ける係数（外側の全検証窓で決定）: {factor_all:.5f}")

        exp = replace(
            base,
            model=f"{base_run_id}+scale",
            hypothesis="H10（1 位解法）: log 学習の予測に 1 未満の係数を掛けると RMSPE が下がる",
            changed_one_thing=(
                f"{base_run_id} の予測に係数を掛ける（係数は外側の各分割の学習期間の中だけで"
                "内側の時系列 CV を組み、その OOF で決める）"
            ),
            params={
                "base_run_id": base_run_id,
                "factor_fit": "inner_time_cv_within_outer_train",
                "factor_for_test": round(factor_all, 5),
            },
        )
        result = CrossFitResult(scaled, factor_all * test_pred, infos)
        inputs = STORE.input_hashes(base_run_id, "oof.npy", "test.npy")
        save_run(reservation, exp, data, splits, result, time.perf_counter() - started, inputs)


if __name__ == "__main__":
    main()
