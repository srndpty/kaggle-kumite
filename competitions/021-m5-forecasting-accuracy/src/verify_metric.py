"""主催者の公開重みと、Kaggle ダミー提出の Public スコアを照合する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import HORIZON, PRIVATE_ORIGIN, PUBLIC_ORIGIN, load_sales
from metric import LEVELS, Hierarchy, Scorer
from model import seasonal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    data = load_sales(evaluation=True)
    hierarchy = Hierarchy.from_series(data.series)
    scorer = Scorer.at(data, hierarchy, PRIVATE_ORIGIN)
    reference = pd.read_csv(args.weights)
    level = reference["Level_id"].str.removeprefix("Level").astype(int)
    first = reference["Agg_Level_1"]
    second = reference["Agg_Level_2"]
    single = level.isin(i for i, keys in enumerate(LEVELS, start=1) if len(keys) == 1)
    name = (first + "_" + second).mask(single, first).mask(level == 11, second + "_" + first)
    weights = reference.assign(level=level, series=name).set_index(["level", "series"])["weight"]
    local = hierarchy.labels.assign(weight=scorer.weight).set_index(["level", "series"])["weight"]
    if (
        not weights.index.is_unique
        or len(weights) != len(local)
        or not local.index.isin(weights.index).all()
    ):
        raise ValueError("公開重みの系列キーが一致しません")
    expected = weights.reindex(local.index).to_numpy()
    np.testing.assert_allclose(local.to_numpy(), expected, rtol=0, atol=1e-8)
    print(f"weights={len(weights)}, max_abs_diff={np.max(np.abs(local.to_numpy() - expected)):.3g}")
    public = Scorer.at(data, hierarchy, PUBLIC_ORIGIN)
    levels = public.score(
        data.actual(PUBLIC_ORIGIN + 1, PUBLIC_ORIGIN + HORIZON),
        seasonal(data, PUBLIC_ORIGIN),
    )
    print(f"dummy_public_wrmsse={levels.mean():.8f}")


if __name__ == "__main__":
    main()
