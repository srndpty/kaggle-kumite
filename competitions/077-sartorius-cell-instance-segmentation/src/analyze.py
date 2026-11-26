"""検証画像の細胞種・密度・面積別に誤差を集計する。"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from data import ROOT, dataset
from masks import image_score, ious, pixels
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from train import PREDICTIONS, assess, read_rows

from kumite.quality_gates import verify_cv_run
from kumite.runs import RunStore


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn


def band(value: int, lower: int, upper: int) -> str:
    if value < lower:
        return f"<{lower}"
    if value < upper:
        return f"{lower}–{upper - 1}"
    return f"{upper}+"


def analyze(run_id: str, output: Path) -> None:
    store = RunStore(ROOT)
    verify_cv_run(store, run_id, assess)
    rows, folds, _, _ = dataset()
    predictions = cast(
        "list[list[str]]", read_rows(store.artifact_path(run_id, PREDICTIONS), rows, folds)
    )
    counts: defaultdict[tuple[str, str], Counts] = defaultdict(Counts)
    scores: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        overlap = ious(row.masks, prediction, row.size)
        matching = maximum_bipartite_matching(csr_matrix(overlap > 0.5), perm_type="column")
        detected = {int(index) for index in matching if index >= 0}
        for key in (
            ("細胞種", row.cell_type),
            ("画像内の正解個体数", band(len(row.masks), 50, 150)),
        ):
            counts[key].add(
                len(detected), len(prediction) - len(detected), len(row.masks) - len(detected)
            )
            scores[key].append(image_score(overlap))
        for rle, hit in zip(row.masks, matching >= 0, strict=True):
            key = ("マスク面積 (px)", band(len(pixels(rle, row.size)), 100, 500))
            counts[key].add(int(hit), 0, int(not hit))
        for index, rle in enumerate(prediction):
            if index not in detected:
                key = ("マスク面積 (px)", band(len(pixels(rle, row.size)), 100, 500))
                counts[key].add(0, 1, 0)
    lines = [
        f"# {run_id} の OOF エラー分析",
        "",
        "TP・FP・FN は IoU > 0.50。面積別の TP・FN は正解マスク、FP は予測マスクで区分した。",
        "",
        "| 分類 | 区分 | 画像数 | 平均 AP | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, count in sorted(counts.items()):
        values = scores[key]
        average = f"{np.mean(values):.5f}" if values else "—"
        size = str(len(values)) if values else "—"
        tp, fp, fn = asdict(count).values()
        lines.append(f"| {key[0]} | {key[1]} | {size} | {average} | {tp} | {fp} | {fn} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.run_id, args.output)
