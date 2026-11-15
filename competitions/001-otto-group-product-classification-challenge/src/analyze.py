"""OOF 予測の誤差分析（クラス別 logloss・混同行列・確率の較正具合）。

実行: uv run python competitions/001-otto-group-product-classification-challenge/src/analyze.py E01
"""

import argparse

import numpy as np
import pandas as pd
from dataset import CLASSES, TARGET, encode_target, load_train
from pipeline import load_predictions

EPS = 1e-15
N_BINS = 10


def per_class_loss(y: np.ndarray, oof: np.ndarray) -> None:
    sample_loss = -np.log(np.clip(oof[np.arange(len(y)), y], EPS, 1))
    total = sample_loss.sum()
    table = pd.DataFrame(
        {
            "件数比": np.bincount(y) / len(y),
            "平均logloss": [sample_loss[y == k].mean() for k in range(len(CLASSES))],
            "損失への寄与": [sample_loss[y == k].sum() / total for k in range(len(CLASSES))],
        },
        index=CLASSES,
    )
    table["寄与/件数比"] = table["損失への寄与"] / table["件数比"]
    print("=== クラス別 logloss（正解クラスごと）===")
    print(table.round(4).to_string())
    group = np.isin(y, [1, 2, 3])  # Class_2, 3, 4
    print(
        f"\nClass_2-4: 件数比 {group.mean():.4f} / 損失への寄与 {sample_loss[group].sum() / total:.4f}"
    )


def confusion(y: np.ndarray, oof: np.ndarray) -> None:
    pred = oof.argmax(axis=1)
    print(
        f"\n=== 混同行列（argmax、行 = 正解、行ごとに正規化）=== accuracy {np.mean(pred == y):.4f}"
    )
    labels = range(len(CLASSES))
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"), normalize="index")
    # 予測に出ないクラス（E00 のように全行同じ予測など）も 0 の行・列として残す
    cm = cm.reindex(index=labels, columns=labels, fill_value=0.0)
    cm.index, cm.columns = CLASSES, CLASSES
    print(cm.round(3).to_string())
    off = cm.to_numpy().copy()
    np.fill_diagonal(off, 0)
    print("\n取り違えの多い組（正解 → 予測、正解クラス内の割合）:")
    for flat in np.argsort(off, axis=None)[::-1][:6]:
        i, j = divmod(int(flat), len(CLASSES))
        print(f"  {CLASSES[i]} → {CLASSES[j]}: {off[i, j]:.3f}")


def reliability(y: np.ndarray, oof: np.ndarray) -> None:
    conf = oof.max(axis=1)
    correct = oof.argmax(axis=1) == y
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    rows = [
        (f"{b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}", m.sum(), conf[m].mean(), correct[m].mean())
        for b in range(N_BINS)
        if (m := bins == b).any()
    ]
    table = pd.DataFrame(rows, columns=["最大確率", "件数", "平均確信度", "正解率"])
    ece = float((table["件数"] * (table["平均確信度"] - table["正解率"]).abs()).sum() / len(y))
    print("\n=== 確率の較正（最大確率のビンごと）===")
    print(table.round(4).to_string(index=False))
    print(f"ECE: {ece:.4f}（平均確信度 > 正解率 なら過信）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    run_id: str = parser.parse_args().run_id
    y = encode_target(load_train()[TARGET]).to_numpy()
    oof, _ = load_predictions(run_id)
    pd.set_option("display.width", 160)
    print(f"run: {run_id}\n")
    per_class_loss(y, oof)
    confusion(y, oof)
    reliability(y, oof)


if __name__ == "__main__":
    main()
