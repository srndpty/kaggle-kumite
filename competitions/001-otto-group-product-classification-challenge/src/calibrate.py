"""既存 run の予測に温度スケーリングをかける後処理実験。

温度 T は OOF からクロスフィットで決める（fold k の T は k 以外の fold の OOF で決める）ので、
OOF スコアに検証データの情報は漏れない。テスト予測には OOF 全体で決めた T を使う。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/calibrate.py E01 E06
"""

import argparse
import time
from typing import cast

import numpy as np
from cv import N_SPLITS, make_folds
from dataset import TARGET, encode_target, load_test, load_train
from pipeline import STORE, Experiment, load_predictions, multi_logloss, save_run
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import softmax

from kumite.crossfit import CrossFitResult
from kumite.runs import Reservation

EPS = 1e-15
T_BOUNDS = (0.2, 5.0)


def apply_temperature(pred: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.log(np.clip(pred, EPS, 1)) / t, axis=1)


def fit_temperature(y: np.ndarray, pred: np.ndarray) -> float:
    # scipy の型情報では戻り値が object なので、実際の型 OptimizeResult を明示する。
    result = cast(
        OptimizeResult,
        minimize_scalar(
            lambda t: multi_logloss(y, apply_temperature(pred, t)),
            bounds=T_BOUNDS,
            method="bounded",
        ),
    )
    return float(result.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run_id")
    parser.add_argument("run_id")
    args = parser.parse_args()
    base_run_id: str = args.base_run_id
    run_id: str = args.run_id
    with STORE.reserve(run_id) as reservation:
        calibrate(base_run_id, reservation)


def calibrate(base_run_id: str, reservation: Reservation) -> None:
    y = encode_target(load_train()[TARGET])
    folds = make_folds(y)
    oof, test_pred = load_predictions(base_run_id)

    started = time.perf_counter()
    calibrated = np.zeros_like(oof)
    fold_infos = []
    for fold in range(N_SPLITS):
        tr, va = folds != fold, folds == fold
        t = fit_temperature(y[tr].to_numpy(), oof[tr])
        calibrated[va] = apply_temperature(oof[va], t)
        fold_infos.append({"temperature": round(t, 4)})
    t_all = fit_temperature(y.to_numpy(), oof)
    runtime = time.perf_counter() - started
    print(f"テスト予測に使う温度: {t_all:.4f}")

    exp = Experiment(
        model=f"{base_run_id}+temperature_scaling",
        hypothesis="H4: 木モデルの確率は較正がずれており、温度スケーリングで logloss が下がる",
        changed_one_thing=f"{base_run_id} の予測に温度スケーリングを適用（T はクロスフィット）",
        params={"base_run_id": base_run_id, "temperature_for_test": round(t_all, 4)},
    )
    result = CrossFitResult(calibrated, apply_temperature(test_pred, t_all), fold_infos)
    inputs = STORE.input_hashes(base_run_id, "oof.npy", "test.npy")
    save_run(reservation, exp, y, folds, result, load_test()["id"], runtime, inputs)


if __name__ == "__main__":
    main()
