"""公式学習データの構成と重複、検証 fold の正例数を集計する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import FOLDS, data_fingerprint, eligible_noisy, read_clips, split_groups, targets

ROOT = Path(__file__).resolve().parents[1]


def summarize(output: Path) -> None:
    root = ROOT / "data"
    tags = list(pd.read_csv(root / "sample_submission.csv", nrows=0).columns[1:])
    curated = read_clips(root, "curated", check_gain=True)
    noisy = read_clips(root, "noisy", check_gain=True)
    y = targets(curated, tags)
    folds = split_groups(curated, y)
    result: dict[str, object] = {"data_hash": data_fingerprint(root, curated + noisy)}
    gain_groups: dict[str, list[tuple[str, str]]] = {}
    for clip in curated + noisy:
        if clip.gain_group is not None:
            gain_groups.setdefault(clip.gain_group, []).append((clip.path.name, clip.group))
    result["gain_or_polarity_matches"] = [
        [name for name, _ in members]
        for members in gain_groups.values()
        if len({digest for _, digest in members}) > 1
    ]
    for name, clips in (("curated", curated), ("noisy", noisy)):
        labels = targets(clips, tags)
        numeric_labels = labels.astype(np.int64)
        cooccurrence = numeric_labels.T @ numeric_labels
        positive = np.triu(cooccurrence, k=1)
        pairs = [(int(positive[i, j]), tags[i], tags[j]) for i, j in np.argwhere(positive)]
        result[name] = {
            "clips": len(clips),
            "groups": len({clip.group for clip in clips}),
            "length_seconds_quantiles": np.percentile(
                [clip.seconds for clip in clips], [0, 25, 50, 75, 90, 100]
            ).tolist(),
            "tag_frequency": dict(zip(tags, labels.sum(axis=0).tolist(), strict=True)),
            "most_common_pairs": sorted(pairs, reverse=True)[:20],
        }
    summaries = []
    for fold in range(FOLDS):
        mask = folds == fold
        validation = [c for c, membership in zip(curated, mask, strict=True) if membership]
        summaries.append(
            {
                "fold": fold,
                "clips": len(validation),
                "tag_frequency": dict(zip(tags, y[mask].sum(axis=0).tolist(), strict=True)),
                "noisy_excluded": len(noisy) - len(eligible_noisy(noisy, validation)),
            }
        )
    result["folds"] = summaries
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.output)
