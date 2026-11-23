"""保存済みの CV 予測から階層・分位点・間欠性別の誤差を集計する。"""

import argparse

import numpy as np
import pandas as pd
from metric import scale_at, score
from model import QUANTILE_NAMES
from train import DATA, ORIGINS, STORE, hierarchy_for

from kumite.m5 import HORIZON, LEVELS, load_sales, revenue_weights


def intervals(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {}
    for percent, lower, upper in ((50, 3, 5), (95, 1, 7)):
        low, high = prediction[..., lower], prediction[..., upper]
        result[f"coverage_{percent}"] = float(((actual >= low) & (actual <= high)).mean())
        result[f"width_{percent}"] = float((high - low).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    options = parser.parse_args()
    data = load_sales(DATA)
    hierarchy = hierarchy_for(data)
    for run_id in options.run_ids:
        rows = []
        for origin in ORIGINS:
            actual = hierarchy.aggregate(data.actual(origin + 1, origin + HORIZON))
            prediction = STORE.load_array(run_id, f"d_{origin}.npy")
            weight = revenue_weights(data, hierarchy, origin)
            scale = scale_at(data, hierarchy, origin)
            losses = score(actual, prediction, hierarchy, weight, scale)
            for level in range(len(LEVELS)):
                mask = hierarchy.levels == level
                rows.append(
                    {
                        "segment": f"level_{level + 1:02d}",
                        "wspl": losses[level].mean(),
                        **dict(zip(QUANTILE_NAMES, losses[level], strict=True)),
                        **intervals(actual[mask], prediction[mask]),
                    }
                )
            zero_rate = (hierarchy.aggregate(data.actual(origin - 335, origin)) == 0).mean(axis=1)
            bottom = hierarchy.levels == len(LEVELS) - 1
            for name, segment in (
                ("zero_ge_50pct", zero_rate >= 0.5),
                ("zero_lt_50pct", zero_rate < 0.5),
            ):
                mask = bottom & segment
                subset_weight = np.where(mask, weight, 0)
                subset_loss = score(actual, prediction, hierarchy, subset_weight, scale)[-1]
                subset_loss /= subset_weight.sum()
                rows.append(
                    {
                        "segment": name,
                        "wspl": subset_loss.mean(),
                        **dict(zip(QUANTILE_NAMES, subset_loss, strict=True)),
                        **intervals(actual[mask], prediction[mask]),
                    }
                )
            for week in range(4):
                days = slice(7 * week, 7 * (week + 1))
                rows.append(
                    {
                        "segment": f"week_{week + 1}",
                        **intervals(actual[:, days], prediction[:, days]),
                    }
                )
        print(run_id)
        print(pd.DataFrame(rows).groupby("segment").mean().round(6).to_string())


if __name__ == "__main__":
    main()
