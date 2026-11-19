"""保存済みのテスト確率から公式形式の提出 CSV を出力する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TEXT = ["question1", "question2"]


def read_test(root: Path) -> tuple[pd.Series, pd.DataFrame]:
    sample = pd.read_csv(root / "data" / "sample_submission.csv")
    if list(sample.columns) != ["test_id", "is_duplicate"]:
        raise ValueError("提出見本の列が不正です")
    ids = sample["test_id"]
    test = pd.read_csv(root / "data" / "test.csv", nrows=len(ids))
    if ids.isna().any() or ids.duplicated().any() or not ids.equals(test["test_id"]):
        raise ValueError("提出見本とテストの ID が一致しません")
    return ids, test


def write_submission(ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    if pred.shape != (len(ids),) or not np.isfinite(pred).all():
        raise ValueError("予測の形状または有限性が不正です")
    if ((pred < 0) | (pred > 1)).any():
        raise ValueError("予測確率が範囲外です")
    pd.DataFrame({"test_id": ids, "is_duplicate": pred}).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    store = RunStore(ROOT)
    ids = pd.read_csv(ROOT / "data" / "sample_submission.csv")["test_id"]
    saved = store.artifact_path(args.run_id, "submission.csv")
    if not ids.equals(pd.read_csv(saved, usecols=["test_id"])["test_id"]):
        raise ValueError("提出見本と保存済み run の ID が一致しません")
    pred = store.load_array(args.run_id, "test.npy")
    write_submission(ids, pred, args.output)


if __name__ == "__main__":
    main()
