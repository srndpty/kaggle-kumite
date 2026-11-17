"""エラー分析と、小さい差の確認（運転手単位の bootstrap）。

実行:
  uv run python competitions/060-state-farm-distracted-driver-detection/src/analyze.py errors E02
  uv run python competitions/060-state-farm-distracted-driver-detection/src/analyze.py compare E02 E03 E04
  uv run python competitions/060-state-farm-distracted-driver-detection/src/analyze.py compare E02 E02R --cross-split

bootstrap は運転手（26 人）を再標本化する。グループが少ないので区間は粗い。
"""

import argparse

import numpy as np
import pandas as pd
from dataset import CLASSES, DRIVER, load_train
from models import row_logloss
from pipeline import STORE_RUNS

from kumite.bootstrap import group_bootstrap_diff
from kumite.experiments import compare_runs

N_BOOTSTRAP = 2000


def _rows(run_id: str) -> pd.DataFrame:
    """学習画像の全行に、OOF の確率から作る行ごとの logloss・予測クラスを付ける（全行が検証済み）。"""
    train = load_train()
    proba = STORE_RUNS.load_array(run_id, "oof.npy")
    if not np.isfinite(proba).all():
        raise ValueError(f"{run_id} の OOF に検証されていない行があります")
    y = train["label"].to_numpy()
    return train.assign(
        loss=row_logloss(y, proba), pred=proba.argmax(axis=1), p_true=proba[np.arange(len(y)), y]
    )


def errors(run_id: str) -> None:
    rows = _rows(run_id)
    print(
        f"{run_id}: logloss {rows['loss'].mean():.5f}、正解率 {(rows['pred'] == rows['label']).mean():.4f}"
    )
    rows = rows.assign(ok=rows["pred"] == rows["label"])
    stats = {"logloss": ("loss", "mean"), "acc": ("ok", "mean"), "rows": ("loss", "size")}
    by_class = rows.groupby("classname").agg(**stats)
    print(f"\nクラスごと:\n{by_class.round(4).to_string()}")
    by_driver = rows.groupby(DRIVER).agg(**stats)
    print(
        f"\n運転手ごと（logloss の悪い順）:\n{by_driver.sort_values('logloss', ascending=False).round(4).to_string()}"
    )
    confusion = pd.crosstab(
        rows["classname"],
        rows["pred"].map(dict(enumerate(CLASSES))),
        rownames=["正解"],
        colnames=["予測"],
    )
    print(f"\n混同行列:\n{confusion.to_string()}")
    wrong = rows[rows["pred"] != rows["label"]]
    pairs = (wrong["classname"] + "→" + wrong["pred"].map(dict(enumerate(CLASSES)))).value_counts()
    print(f"\n取り違えの多い組（上位 8）:\n{pairs.head(8).to_string()}")
    loss = rows["loss"].sort_values(ascending=False)
    for share in (0.01, 0.05):
        print(
            f"logloss の大きい上位 {share:.0%} の行が損失の合計に占める割合: {loss.head(int(len(loss) * share)).sum() / loss.sum():.3f}"
        )


def compare(base_run_id: str, run_ids: list[str], cross_split: bool) -> None:
    """base に対する各 run の logloss の差（other - base。負が改善）を、運転手単位の bootstrap で確かめる。

    cross_split=True なら分割の違う run どうしも比べる（同じ全行を評価しているので、行は対応する）。
    """
    base = _rows(base_run_id)
    groups = base[DRIVER].to_numpy()
    print(
        f"基準 {base_run_id}: logloss {base['loss'].mean():.5f}（運転手 {len(np.unique(groups))} 人を再標本化）"
    )
    for run_id in run_ids:
        if not cross_split:
            compare_runs(STORE_RUNS.record(base_run_id), STORE_RUNS.record(run_id))
        other = _rows(run_id)
        diff = group_bootstrap_diff(
            base["loss"].to_numpy(),
            other["loss"].to_numpy(),
            groups,
            transform=lambda m: m,
            n_bootstrap=N_BOOTSTRAP,
        )
        better = (other.groupby(DRIVER)["loss"].mean() < base.groupby(DRIVER)["loss"].mean()).sum()
        print(
            f"  {run_id}: {other['loss'].mean():.5f}、差 {diff.point:+.5f}"
            f"（95% 区間 [{diff.low:+.5f}, {diff.high:+.5f}]、改善の割合 {diff.share_negative:.3f}、"
            f"改善した運転手 {better} / {len(np.unique(groups))}）"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_err = sub.add_parser("errors")
    p_err.add_argument("run_id")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("base")
    p_cmp.add_argument("run_ids", nargs="+")
    p_cmp.add_argument("--cross-split", action="store_true")
    args = parser.parse_args()
    if args.command == "errors":
        errors(args.run_id)
    else:
        compare(args.base, args.run_ids, args.cross_split)


if __name__ == "__main__":
    main()
