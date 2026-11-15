"""H6: 提出の作り方の比較。train の中で提出と同じ状況を再現する。

最後の検証窓（分割 2）を疑似テストとして取り置き、それより前の行だけで次の 2 通りを作って比べる。

- 分割平均: 手前の 2 窓（分割 0・1）で early stopping したモデル 2 つの平均（提出の既定の形）
- 学習し直し: 疑似テストより前の全行で 1 モデルを学習する。round 数は上の 2 モデルの best iteration の平均

実行: uv run python competitions/019-rossmann-store-sales/src/refit_check.py E03
"""

import argparse
from dataclasses import replace

import numpy as np
from pipeline import Data, Experiment, fit_split, load_data, refit, rmspe
from train import EXPERIMENTS


def check(exp: Experiment, data: Data) -> None:
    splits = data.splits("time")
    y = data.y
    (inner, (refit_train, pseudo)) = splits[:-1], splits[-1]
    # 疑似テストの行を「テスト」として渡す（分割 0・1 の学習行・検証行はどちらも疑似テストより前）
    held = replace(data, test=data.train.iloc[pseudo], x_test=data.x_train.iloc[pseudo])
    y_pseudo = y[pseudo]

    preds, rounds = [], []
    for k, (tr, va) in enumerate(inner):
        out = fit_split(exp, held, y, tr, va)
        preds.append(out.query)
        rounds.append(out.info["best_iteration"])
        print(
            f"分割 {k} のモデル: 検証窓 {rmspe(y[va], out.valid):.5f}、"
            f"疑似テスト {rmspe(y_pseudo, out.query):.5f}（best iteration {rounds[-1]}）",
            flush=True,
        )
    print(f"分割平均（2 モデル）: 疑似テスト {rmspe(y_pseudo, np.mean(preds, axis=0)):.5f}")

    n_rounds = round(float(np.mean(rounds)))
    refitted = refit(exp, held, y, refit_train, pseudo, n_rounds)
    print(
        f"学習し直し（{n_rounds} round、検証行は見ない）: 疑似テスト {rmspe(y_pseudo, refitted.valid):.5f}"
    )
    # 参考: 分割 1 と同じ学習行（疑似テストの 48 日前まで）のまま、round 数だけそろえた場合
    last_tr, _ = inner[-1]
    same_rows = refit(exp, held, y, last_tr, pseudo, n_rounds)
    print(
        f"参考・分割 1 と同じ学習行で {n_rounds} round: 疑似テスト {rmspe(y_pseudo, same_rows.valid):.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", choices=list(EXPERIMENTS))
    data = load_data()
    for run_id in parser.parse_args().run_ids:
        print(f"=== {run_id} ===")
        check(EXPERIMENTS[run_id], data)


if __name__ == "__main__":
    main()
