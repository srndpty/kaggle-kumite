"""Kaggle の採点可否を公式のゼロ予測で確認する。"""

from pathlib import Path

import numpy as np
from model import QUANTILES
from submission import write_submission

from kumite.m5 import HORIZON, Hierarchy, load_sales

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    hierarchy = Hierarchy.from_series(load_sales(DATA).series)
    zeros = np.zeros((len(hierarchy.levels), HORIZON, len(QUANTILES)), dtype=np.uint8)
    output = ROOT / "outputs" / "dummy.csv"
    output.parent.mkdir(exist_ok=True)
    write_submission(output, DATA / "sample_submission.csv", (hierarchy, zeros), (hierarchy, zeros))
    print(output)


if __name__ == "__main__":
    main()
