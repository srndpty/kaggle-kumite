"""公開済みの Private 正解による外部評価。

学習・選定・export のコード（`SOURCE_FILES`）からは読み込まない。テストで import しないことを確かめる。
P4〜P6 の開発中は使わず、最終選定記録を確定した後に `analyze.py` から実行する。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data import DATA, SCORED, TRAIN_SCORED, mcrmse

PRIVATE_LABELS = DATA / "post_deadline_files" / "private_test_labels.csv"
PRIVATE_SCORED = 91  # P0 で Kaggle の Private スコア（全 0 提出 0.71003）と一致した採点位置数


def private_truth(labels: Path = PRIVATE_LABELS) -> tuple[list[str], np.ndarray]:
    """採点対象（`test_filter == 1`）の配列 ID と、(配列, 91 位置, 採点 3 列) の正解。"""
    truth = pd.read_csv(labels)
    truth = truth[truth["test_filter"] == 1]
    y = np.stack(
        [np.array([json.loads(str(v))[:PRIVATE_SCORED] for v in truth[c]]) for c in SCORED], -1
    )
    return list(truth["id"]), y


def private_breakdown(submission: Path, labels: Path = PRIVATE_LABELS) -> dict[str, float]:
    """提出 CSV の Private MCRMSE（全体・学習で見た位置 0〜67・外挿位置 68〜90）。"""
    ids, y = private_truth(labels)
    frame = pd.read_csv(submission).set_index("id_seqpos")
    keys = [f"{i}_{p}" for i in ids for p in range(PRIVATE_SCORED)]
    predicted = frame.loc[keys, list(SCORED)].to_numpy(dtype=np.float64).reshape(y.shape)
    return {
        "all": mcrmse(y, predicted),
        "seen_0_67": mcrmse(y[:, :TRAIN_SCORED], predicted[:, :TRAIN_SCORED]),
        "extrapolated_68_90": mcrmse(y[:, TRAIN_SCORED:], predicted[:, TRAIN_SCORED:]),
    }
