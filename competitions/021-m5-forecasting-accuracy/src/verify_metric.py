"""主催者の公開重みと、Kaggle ダミー提出の Public スコアを照合する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import DATA, HORIZON, PRIVATE_ORIGIN, PUBLIC_ORIGIN, Sales, load_sales
from metric import LEVELS, Hierarchy, Scorer
from model import seasonal
from train import STORE


def read_public_csv(path: Path, data: Sales) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = [f"F{day}" for day in range(1, HORIZON + 1)]
    sample = pd.read_csv(DATA / "sample_submission.csv", usecols=["id"])
    if list(frame.columns) != ["id", *columns] or not frame["id"].equals(sample["id"]):
        raise ValueError(f"{path}: 提出 CSV の列または ID・行順が一致しません")
    ids = data.series["item_id"] + "_" + data.series["store_id"] + "_validation"
    return frame.set_index("id").loc[pd.Index(ids), columns].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--dummy-csv", type=Path)
    parser.add_argument("--final-run")
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
    actual = data.actual(PUBLIC_ORIGIN + 1, PUBLIC_ORIGIN + HORIZON)
    dummy = seasonal(data, PUBLIC_ORIGIN)
    levels = public.score(actual, dummy)
    print(f"dummy_public_wrmsse={levels.mean():.8f}")
    pairs = []
    if args.dummy_csv:
        pairs.append((args.dummy_csv, dummy))
    if args.final_run:
        pairs.append(
            (
                STORE.artifact_path(args.final_run, "submission.csv"),
                STORE.load_array(args.final_run, "public_predictions.npy"),
            )
        )
    for path, before in pairs:
        after = read_public_csv(path, data)
        before_score = public.score(actual, before).mean()
        after_score = public.score(actual, after).mean()
        print(
            f"{path.name}: before={before_score:.10f}, after={after_score:.10f}, "
            f"delta={after_score - before_score:+.3g}, "
            f"max_prediction_diff={np.max(np.abs(before - after)):.3g}"
        )


if __name__ == "__main__":
    main()
