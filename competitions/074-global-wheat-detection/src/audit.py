"""撮影元ごとの分布と近重複候補を確認する。"""

import json
from collections import defaultdict
from itertools import combinations

import numpy as np
from dataset import ROOT, load_samples, split_images
from PIL import Image


def main() -> None:
    samples, excluded = load_samples(ROOT / "data")
    assignments = split_images(samples, "source-v1")
    groups: dict[str, list[int]] = defaultdict(list)
    areas: dict[str, list[float]] = defaultdict(list)
    hashes: list[int] = []
    for sample in samples:
        groups[sample.source].append(len(sample.boxes))
        areas[sample.source].extend((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in sample.boxes)
        with Image.open(sample.path) as image:
            pixels = np.asarray(image.convert("L").resize((9, 8)))
        bits = (pixels[:, :-1] > pixels[:, 1:]).reshape(-1)
        hashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    candidates = [
        {
            "images": [samples[i].path.stem, samples[j].path.stem],
            "sources": [samples[i].source, samples[j].source],
            "distance": distance,
        }
        for i, j in combinations(range(len(samples)), 2)
        if (distance := (hashes[i] ^ hashes[j]).bit_count()) <= 4
    ]
    print(
        json.dumps(
            {
                "excluded": excluded,
                "fold_sizes": [assignments.count(fold) for fold in range(3)],
                "source_summary": {
                    source: {
                        "images": len(counts),
                        "median_boxes_per_image": float(np.median(counts)),
                        "median_box_area": float(np.median(areas[source]))
                        if areas[source]
                        else None,
                        "empty_images": counts.count(0),
                    }
                    for source, counts in sorted(groups.items())
                },
                "near_duplicate_candidates": candidates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
