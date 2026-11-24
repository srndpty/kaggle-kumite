"""OOF の候補漏れと誤一致を、出品の属性別に集計する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from train import ROOT

from kumite.runs import RunStore


def annotate_pairs(path: Path, labels: pd.Series) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    pairs["correct"] = pairs["posting_id"].map(labels) == pairs["candidate_id"].map(labels)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = RunStore(ROOT).runs_dir
    rows = pd.read_csv(ROOT / "data/train.csv").set_index("posting_id")
    labels = rows["label_group"]
    sizes = labels.map(labels.value_counts())
    title = annotate_pairs(runs / "E00/candidates.csv.gz", labels)
    best_title = (
        title.loc[title["correct"]].groupby("posting_id")["score"].max().reindex(rows.index)
    )
    pairs = annotate_pairs(runs / args.run_id / "candidates.csv.gz", labels)
    pairs["false_match"] = pairs["accepted"] & ~pairs["correct"]
    pairs["missed_match"] = pairs["correct"] & ~pairs["accepted"]
    counts = pairs.groupby("posting_id")[["correct", "false_match", "missed_match"]].sum()
    annotated = pd.read_csv(runs / args.run_id / "oof.csv").set_index("posting_id").join(counts)
    annotated["outside_candidates"] = sizes - 1 - annotated["correct"]
    bins = {
        "group_size": pd.cut(sizes, [0, 2, 5, np.inf], labels=["1–2", "3–5", "6+"]),
        "title_cosine": pd.cut(
            best_title, [-0.01, 0.5, 0.8, 1.001], labels=["≤0.5", "0.5–0.8", ">0.8"]
        ),
        "shared_image": rows["image"].duplicated(keep=False),
    }
    summaries: list[pd.DataFrame] = []
    for name, groups in bins.items():
        summaries.append(
            annotated.groupby(groups, observed=True, dropna=False)
            .agg(
                rows=("f1", "size"),
                f1=("f1", "mean"),
                outside_candidates=("outside_candidates", "mean"),
                rejected_matches=("missed_match", "mean"),
                false_matches=("false_match", "mean"),
            )
            .rename_axis("value")
            .reset_index()
            .assign(slice=name)
        )
    pd.concat(summaries, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
