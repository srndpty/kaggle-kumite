"""既存 run の予測に温度スケーリングをかける後処理実験（nested）。

外側 fold k の温度 T は、外側の学習行だけで元の実験を内側 4 fold で学習し直した OOF から決め、
元の run の検証行の予測（外側の学習行だけで学習した fold モデルの予測）にかける。
保存済みの全行 OOF の学習行部分で T を決めると、その OOF を作ったモデルが検証行のラベルでも
学習しているため、検証ラベルが T に漏れる（旧実装の E06）。
テスト予測には、元の run の全行 OOF で決めた T をかける（CV の評価には使わない）。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/calibrate.py E01 E17
"""

import argparse
import time

import numpy as np
from cv import N_SPLITS, make_folds, make_inner_folds
from dataset import TARGET, encode_target, load_test, load_train
from pipeline import STORE, Experiment, load_predictions, multi_logloss, run_cv, save_run
from train import EXPERIMENTS

from kumite.calibration import apply_temperature, fit_temperature
from kumite.crossfit import CrossFitResult
from kumite.runs import Reservation


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
    train, test = load_train(), load_test()
    y = encode_target(train[TARGET])
    folds = make_folds(y)
    oof, test_pred = load_predictions(base_run_id)
    base = EXPERIMENTS[base_run_id]

    started = time.perf_counter()
    calibrated = np.zeros_like(oof)
    fold_infos = []
    for fold in range(N_SPLITS):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        y_tr = y.iloc[tr]
        inner = run_cv(base, train.iloc[tr], y_tr, make_inner_folds(y_tr), train.iloc[va])
        t = fit_temperature(y_tr.to_numpy(), inner.oof, multi_logloss)
        calibrated[va] = apply_temperature(oof[va], t)
        fold_infos.append({"temperature": round(t, 4)})
        print(f"fold {fold}: T = {t:.4f}", flush=True)
    t_all = fit_temperature(y.to_numpy(), oof, multi_logloss)
    runtime = time.perf_counter() - started
    print(f"テスト予測に使う温度: {t_all:.4f}")

    exp = Experiment(
        model=f"{base_run_id}+temperature_scaling",
        hypothesis="H4: 木モデルの確率は較正がずれており、温度スケーリングで logloss が下がる",
        changed_one_thing=f"{base_run_id} の予測に温度スケーリングを適用（T は外側の学習行だけで作った内側 OOF で決める）",
        params={"base_run_id": base_run_id, "temperature_for_test": round(t_all, 4)},
    )
    result = CrossFitResult(calibrated, apply_temperature(test_pred, t_all), fold_infos)
    inputs = STORE.input_hashes(base_run_id, "oof.npy", "test.npy")
    save_run(reservation, exp, y, folds, result, test["id"], runtime, inputs)


if __name__ == "__main__":
    main()
