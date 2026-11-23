"""P0 のために、公式雛形と同じ ID 順でゼロ予測を作る。"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ("201610", "201611", "201612", "201710", "201711", "201712")


def make_dummy(sample: pd.DataFrame) -> pd.DataFrame:
    if list(sample.columns) != ["ParcelId", *MONTHS]:
        raise ValueError("公式雛形の提出列と一致しません")
    ids = sample["ParcelId"]
    if ids.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError("提出 ID が空、欠損、または重複しています")
    return sample.assign(**dict.fromkeys(MONTHS, 0))


def main() -> None:
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv", usecols=["ParcelId", *MONTHS])
    path = ROOT / "outputs" / "p0-dummy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_dummy(sample).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
