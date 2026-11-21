"""CV でモデルを固定し、ホールドアウトの誤差を集計する。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from model import average_precision
from train import ROOT, SEED

from kumite.bootstrap import group_bootstrap_diff
from kumite.provenance import file_sha256
from kumite.runs import RunStore

CANDIDATES = {
    "E01": "baseline",
    "E02": "recent",
    "E03": "portfolio",
    "E04": "logistic",
    "E05": "seasonal",
}


def select() -> None:
    store = RunStore(ROOT)
    records = {key: store.record(key) for key in CANDIDATES}
    baseline = records["E01"]
    for key, variant in CANDIDATES.items():
        if (
            any(
                getattr(records[key], field) != getattr(baseline, field)
                for field in ("split_hash", "data_hash", "split_version")
            )
            or records[key].params["variant"] != variant
        ):
            raise ValueError(f"比較条件が一致しません: {key}")
    best = max(CANDIDATES, key=lambda key: records[key].oof_score)
    selected = best
    interval = None
    if best != "E01":
        base_file = store.runs_dir / "E01" / "evaluation.npz"
        other_file = store.runs_dir / best / "evaluation.npz"
        for key, path in (("E01", base_file), (best, other_file)):
            if file_sha256(path) != records[key].artifacts[path.name]:
                raise ValueError(f"検証成果物のハッシュが一致しません: {key}")
        with (
            np.load(base_file) as base,
            np.load(other_file) as other,
        ):
            for field in ("ids", "labels", "months"):
                if not np.array_equal(base[field], other[field]):
                    raise ValueError(f"検証行が一致しません: {field}")
            interval = group_bootstrap_diff(
                average_precision(base["labels"], base["ranked"]),
                average_precision(other["labels"], other["ranked"]),
                base["ids"],
                transform=lambda ap: ap,
                n_bootstrap=1000,
                seed=SEED,
            )
        if interval.low <= 0:
            selected = "E01"
    selection = {
        "selected_run_id": selected,
        "variant": CANDIDATES[selected],
        "highest_cv_run_id": best,
        "cv_map_at_7": {key: records[key].oof_score for key in CANDIDATES},
        "bootstrap_best_minus_baseline": asdict(interval) if interval else None,
    }
    path = ROOT / "selection.json"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != selection:
            raise ValueError("計算結果が既存の selection.json と一致しません") from None
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def summarize(path: Path) -> None:
    with np.load(path) as data:
        labels, ranked, owned = (data[field] for field in ("labels", "ranked", "owned"))
        ap = average_precision(labels, ranked)
        hit = np.zeros(labels.shape, dtype=bool)
        rows, positions = np.nonzero(ranked >= 0)
        hit[rows, ranked[rows, positions]] = True
        count = np.minimum((owned == 1).sum(axis=1), 2)
        for month in np.unique(data["months"]):
            block = data["months"] == month
            print(
                f"{month} rows={block.sum()} no_addition={float((labels[block] == 0).all(axis=1).mean()):.6f}"
            )
            for group in np.unique(count[block]):
                subset = block & (count == group)
                print(f"owned={group} rows={subset.sum()} map_at_7={ap[subset].mean():.6f}")
        missed = ((labels == 1) & ~hit).sum(axis=0)
        for p in np.argsort(-missed)[:5]:
            print(f"{data['products'][p]} missed={missed[p]} actual={(labels[:, p] == 1).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        select()


if __name__ == "__main__":
    main()
