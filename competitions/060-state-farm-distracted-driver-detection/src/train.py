"""実験の実行。CV で OOF を作り、テスト予測・提出ファイル・実験ログを出力する。

実行: uv run python competitions/060-state-farm-distracted-driver-detection/src/train.py E01
"""

import argparse
import time
from dataclasses import replace

from models import ModelConfig
from pipeline import STORE_RUNS, Experiment, load_data, run_cv, save_run

EXPERIMENTS: dict[str, Experiment] = {
    "E00": Experiment(
        model="prior",
        hypothesis="学習 fold のクラス比を全画像に出す。全実験の上限（これより悪ければ壊れている）",
        changed_one_thing="なし（統計的 baseline）",
    ),
    "E01": Experiment(
        model="linear_probe",
        hypothesis="凍結した ImageNet 学習済み ResNet50 の特徴 + ロジスティック回帰。fine-tune との比較の基準（H2）",
        changed_one_thing="E00 からモデルを、凍結特徴 + ロジスティック回帰（C=0.1）に変更",
        params={"C": 0.1},
    ),
    "E02": Experiment(
        model="finetune",
        hypothesis="H2: ResNet50 を全層 fine-tune すると、凍結特徴 + ロジスティック回帰より大きく良い",
        changed_one_thing="E01 から、backbone も学習する fine-tune に変更（5 epoch、弱い augmentation、反転なし）",
    ),
    "E03": Experiment(
        model="finetune",
        hypothesis="H3: 弱い augmentation を外すと、26 人の運転手に過学習して悪化する",
        changed_one_thing="E02 の augmentation を無しに変更",
        config=ModelConfig(aug="none"),
    ),
    "E04": Experiment(
        model="finetune",
        hypothesis="H3: 左右反転を足すと、右手・左手のクラス（c1/c3、c2/c4）を取り違えて悪化する",
        changed_one_thing="E02 に左右反転（確率 0.5、ラベルはそのまま）を追加",
        config=ModelConfig(hflip=True),
    ),
}
# seed の揺らぎ: 同じ設定を seed だけ変えて回す（小さい差の判断基準）
for _seed in (1, 2):
    EXPERIMENTS[f"E02-s{_seed}"] = replace(
        EXPERIMENTS["E02"],
        hypothesis="seed だけを変えた E02。run 間の差の大きさ（揺らぎ）を測る",
        changed_one_thing=f"E02 の seed を {_seed} に変更（重みの初期化・順序・augmentation の乱数）",
        config=ModelConfig(seed=_seed),
    )
# H1: 同じ実験を、画像単位のランダム分割で測る（run_id に R を付ける）
for _base in ("E01", "E02", "E03"):
    EXPERIMENTS[f"{_base}R"] = replace(
        EXPERIMENTS[_base],
        hypothesis=f"H1: 画像単位のランダム分割は、運転手単位の分割より大きく楽観的（{_base} と同じモデル）",
        changed_one_thing=f"{_base} の検証分割を、同じ全行の画像単位のランダム分割に変更",
        split="random",
    )


def run(run_id: str) -> None:
    exp = EXPERIMENTS[run_id]
    with STORE_RUNS.reserve(run_id) as reservation:  # 学習前に run_id を予約する
        data = load_data()
        splits = data.splits(exp.split)
        started = time.perf_counter()
        result = run_cv(exp, data, splits)
        save_run(reservation, exp, data, splits, result, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    for run_id in parser.parse_args().run_ids:
        run(run_id)


if __name__ == "__main__":
    main()
