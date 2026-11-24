"""P0 の中央値予測を公式サンプル順で保存する。"""

from pathlib import Path

import numpy as np
import pandas as pd

from kumite.checks import validate_regression

ROOT = Path(__file__).resolve().parents[1]


def submission(sample: pd.DataFrame, test_ids: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    if list(sample.columns) != ["id", "loss"]:
        raise ValueError("提出列が不正です")
    ids = sample["id"]
    for index in (ids, test_ids):
        if index.empty or index.hasnans or not index.is_unique:
            raise ValueError("提出 ID が空、欠損、または重複しています")
    if set(ids) != set(test_ids):
        raise ValueError("提出サンプルとテストの ID が一致しません")
    validate_regression(pred, len(test_ids), lower=0)
    return sample.assign(loss=pd.Series(pred, index=test_ids).loc[ids].to_numpy())


def main() -> None:
    data = ROOT / "data"
    sample = pd.read_csv(data / "sample_submission.csv")
    test_ids = pd.read_csv(data / "test.csv", usecols=["id"])["id"]
    loss = pd.read_csv(data / "train.csv", usecols=["loss"])["loss"].to_numpy()
    validate_regression(loss, len(loss), lower=0)
    frame = submission(sample, test_ids, np.full(len(test_ids), np.median(loss)))
    path = ROOT / "outputs" / "p0-dummy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
