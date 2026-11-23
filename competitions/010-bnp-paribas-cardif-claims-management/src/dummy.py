"""公式サンプルと同じ ID 順で P0 用の定数確率を出力する。"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_dummy(sample: pd.DataFrame) -> pd.DataFrame:
    """ID を検査し、全行の予測確率を 0.5 に設定する。"""
    if list(sample.columns) != ["ID", "PredictedProb"]:
        raise ValueError("公式サンプルの提出列と一致しません")
    ids = sample["ID"]
    if sample.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError("提出 ID が空、欠損、または重複しています")
    return sample.assign(PredictedProb=0.5)


def main() -> None:
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    submission = make_dummy(sample)
    path = ROOT / "outputs" / "p0-dummy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
