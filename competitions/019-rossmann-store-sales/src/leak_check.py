"""ラベルを使う特徴が、学習行より後の売上に依存しないことを確かめる（label perturbation test）。

分割 0 の学習行以外（検証窓とそれ以降）の売上をすべて別の値に変えても、学習・検証・テストの入力が
変わらないことを検査する。自分の売上が学習行の特徴に入ること（E05）は、この検査の対象ではない。

実行: uv run python competitions/019-rossmann-store-sales/src/leak_check.py E04 E05
"""

import argparse

import numpy as np
from pipeline import Experiment, load_data
from pipeline import fold_inputs as build_inputs
from train import EXPERIMENTS

from kumite.checks import check_label_independence, perturb_targets


def check(exp: Experiment) -> int:
    """検査した「隠すべき行」の数を返す。依存があれば AssertionError。"""
    data = load_data()
    train_idx, valid_idx = data.splits(exp.split)[0]
    hidden = np.ones(len(data.y), dtype=bool)
    hidden[train_idx] = False

    def build(y: np.ndarray) -> list[np.ndarray]:
        frames = build_inputs(exp, data, y, train_idx, valid_idx)
        return [x.to_numpy(dtype=np.float64, na_value=np.nan) for x in frames]

    # NaN どうしは等しいとみなす（学習期間に無いキーの統計量は NaN になる）
    def build_filled(y: np.ndarray) -> list[np.ndarray]:
        return [np.nan_to_num(a, nan=-1.0) for a in build(y)]

    check_label_independence(build_filled, data.y, perturb_targets(data.y, hidden))
    return int(hidden.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    for run_id in parser.parse_args().run_ids:
        n_hidden = check(EXPERIMENTS[run_id])
        print(f"{run_id}: OK（学習行以外の {n_hidden} 行の売上を変えても入力は不変）")


if __name__ == "__main__":
    main()
