"""固定 CV 範囲の売上・価格・曜日と、階層ごとの構成を集計する。"""

import json

import numpy as np
from dataset import HORIZON, ORIGINS, load_sales
from metric import LEVELS, Hierarchy, Scorer


def main() -> None:
    data = load_sales()
    origin = ORIGINS[-1]
    history = data.actual(1, origin)
    first = np.zeros(len(history), dtype=int)
    sold = (history > 0).any(axis=1)
    first[sold] = (history[sold] > 0).argmax(axis=1)
    active_days = origin - first[sold]
    post_debut_zeros = active_days.sum() - np.count_nonzero(history)
    hierarchy = Hierarchy.from_series(data.series)
    recent = data.actual(origin - HORIZON + 1, origin)
    prices = data.price(origin - HORIZON + 1, origin)
    revenue = np.sum(recent * np.nan_to_num(prices, nan=0), axis=1)
    scorer = Scorer.at(data, hierarchy, origin)
    aggregate_revenue = hierarchy.aggregate(revenue)
    weekday = data.calendar["weekday"].to_numpy()[:origin]
    weekday_mean = {day: float(history[:, weekday == day].mean()) for day in np.unique(weekday)}
    result = {
        "origin": origin,
        "series": len(history),
        "never_sold": int((~sold).sum()),
        "first_sale_day_quantiles": np.quantile(first[sold] + 1, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "overall_zero_rate": float(1 - np.count_nonzero(history) / history.size),
        "zero_rate_after_first_sale": float(post_debut_zeros / active_days.sum()),
        "recent_missing_price_rate": float(np.isnan(prices).mean()),
        "recent_missing_price_with_positive_sales": int(((recent > 0) & np.isnan(prices)).sum()),
        "recent_zero_sale_with_price_rate": float(((recent == 0) & np.isfinite(prices)).mean()),
        "mean_units_by_weekday": weekday_mean,
        "levels": [
            {
                "level": i + 1,
                "keys": keys,
                "series": int(mask.sum()),
                "recent_revenue": float(aggregate_revenue[mask].sum()),
                "max_weighted_series_share": float(scorer.weight[mask].max()),
            }
            for i, keys in enumerate(LEVELS)
            if (mask := hierarchy.levels == i).any()
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
