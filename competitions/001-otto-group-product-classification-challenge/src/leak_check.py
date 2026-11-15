"""label perturbation test: 外側の検証行のラベルだけを別のクラスに変えても、その fold の
学習・検証・テストの入力特徴が変わらないことを確かめる。変われば、検証ラベルが特徴に漏れている。

対象は入力特徴まで。LightGBM は検証 fold で early stopping するので、学習済みモデルは検証ラベルに
依存する（全実験で共通の既知の偏り。P5 で +0.0020 程度と実測）。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/leak_check.py E11 E12
"""

import argparse

import numpy as np
import pandas as pd
from cv import make_folds
from dataset import CLASSES, TARGET, encode_target, load_test, load_train
from pipeline import fold_inputs
from train import EXPERIMENTS

from kumite.checks import check_label_independence, shift_labels


def check(run_id: str, fold: int) -> None:
    exp = EXPERIMENTS[run_id]
    train, test = load_train(), load_test()
    y = encode_target(train[TARGET])
    hidden = make_folds(y) == fold

    def build(labels: np.ndarray) -> list[np.ndarray]:
        x_tr, x_va, x_te = fold_inputs(
            exp,
            train,
            pd.Series(labels, index=y.index),
            np.flatnonzero(~hidden),
            np.flatnonzero(hidden),
            test,
        )
        return [x_tr.to_numpy(), x_va.to_numpy(), x_te.to_numpy()]

    y_arr = y.to_numpy()
    check_label_independence(build, y_arr, shift_labels(y_arr, hidden, len(CLASSES)))
    print(
        f"{run_id}: fold {fold} の検証ラベル {hidden.sum()} 件を変えても、入力特徴は不変",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    run_ids: list[str] = args.run_ids
    for run_id in run_ids:
        check(run_id, args.fold)


if __name__ == "__main__":
    main()
