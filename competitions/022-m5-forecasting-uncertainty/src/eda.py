"""起点時点の売上規模と間欠性を階層別に確認する。"""

import numpy as np
from dummy import DATA

from kumite.m5 import LEVELS, ORIGINS, Hierarchy, load_sales


def main() -> None:
    data = load_sales(DATA)
    hierarchy = Hierarchy.from_series(data.series)
    origin = ORIGINS[0]
    history = hierarchy.aggregate(data.actual(origin - 335, origin))
    for level in range(len(LEVELS)):
        part = history[hierarchy.levels == level]
        print(
            f"{level + 1:2d}  n={len(part):5d}  "
            f"zero={np.median((part == 0).mean(axis=1)):.4f}  "
            f"daily_mean={np.median(part.mean(axis=1)):.4f}"
        )


if __name__ == "__main__":
    main()
