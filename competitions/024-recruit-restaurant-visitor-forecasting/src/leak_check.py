"""分割ごとに作る特徴が、予測の起点より後の情報に依存しないことを確かめる。

分割 0 について 2 つを検査する。

- 来店数: 学習行以外（検証窓とそれ以降）の来店数をすべて別の値に変えても、入力が変わらない
- 予約: 検証窓の開始以後に入った予約の人数を変えても、入力が変わらない。打ち切らない設定
  （E05L）では検証行の入力が変わることも確かめる（検査が効いていることの確認）

実行: uv run python competitions/024-recruit-restaurant-visitor-forecasting/src/leak_check.py E03 E04 E05
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from dataset import DATE
from pipeline import Data, Experiment, load_data
from pipeline import fold_inputs as build_inputs
from train import EXPERIMENTS

from kumite.checks import check_label_independence, perturb_targets

PARTS = ("学習", "検証", "テスト")


def _arrays(frames: tuple[pd.DataFrame, ...]) -> list[np.ndarray]:
    # NaN どうしは等しいとみなす（学習期間に無いキーの統計量は NaN になる）
    return [np.nan_to_num(x.to_numpy(dtype=np.float64, na_value=np.nan), nan=-1.0) for x in frames]


def check_labels(exp: Experiment, data: Data) -> int:
    train_idx, valid_idx = data.splits(exp.split)[0]
    hidden = np.ones(len(data.y), dtype=bool)
    hidden[train_idx] = False

    def build(y: np.ndarray) -> list[np.ndarray]:
        return _arrays(build_inputs(exp, data, y, train_idx, valid_idx))

    check_label_independence(build, data.y, perturb_targets(data.y, hidden))
    return int(hidden.sum())


def changed_by_late_reserves(exp: Experiment, data: Data) -> list[str]:
    """窓の開始以後に入った予約を変えたとき、入力が変わった部分（学習・検証・テスト）の名前。"""
    train_idx, valid_idx = data.splits(exp.split)[0]
    origin = data.train[DATE].iloc[valid_idx].min()
    late = (data.reserves["reserve_datetime"] >= origin).to_numpy()
    if not late.any():
        raise ValueError("窓の開始以後に入った予約がありません")
    visitors = data.reserves["reserve_visitors"].to_numpy()
    perturbed = data.reserves.assign(reserve_visitors=np.where(late, visitors + 7, visitors))
    base = _arrays(build_inputs(exp, data, data.y, train_idx, valid_idx))
    other = _arrays(
        build_inputs(exp, replace(data, reserves=perturbed), data.y, train_idx, valid_idx)
    )
    return [name for name, a, b in zip(PARTS, base, other, strict=True) if not np.array_equal(a, b)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    data = load_data()
    for run_id in parser.parse_args().run_ids:
        exp = EXPERIMENTS[run_id]
        n_hidden = check_labels(exp, data)
        changed = changed_by_late_reserves(exp, data)
        print(f"{run_id}: 来店数 OK（学習行以外の {n_hidden} 行を変えても入力は不変）")
        print(f"{run_id}: 窓の開始以後の予約を変えて入力が変わった部分: {changed or 'なし'}")


if __name__ == "__main__":
    main()
