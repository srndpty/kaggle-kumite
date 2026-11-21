"""CV でモデルを固定し、ホールドアウトの誤差を集計する。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from model import average_precision
from train import EVALUATION_FILE, ROOT, SEED

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


def select(path: Path, suffix: str) -> None:
    store = RunStore(ROOT)
    candidates = {key + suffix: variant for key, variant in CANDIDATES.items()}
    records = {key: store.record(key) for key in candidates}
    baseline = records["E01" + suffix]
    reference = None
    scores: dict[str, float] = {}
    per_row: dict[str, np.ndarray] = {}
    for key, variant in candidates.items():
        if (
            any(
                getattr(records[key], field) != getattr(baseline, field)
                for field in ("split_hash", "data_hash", "split_version")
            )
            or records[key].params["variant"] != variant
        ):
            raise ValueError(f"比較条件が一致しません: {key}")
        artifact = store.runs_dir / key / EVALUATION_FILE
        if file_sha256(artifact) != records[key].artifacts[EVALUATION_FILE]:
            raise ValueError(f"検証成果物のハッシュが一致しません: {key}")
        with np.load(artifact) as data:
            ids, labels, months = (data[field] for field in ("ids", "labels", "months"))
            rows = (ids, labels, months)
            if reference is None:
                reference = rows
            elif not all(np.array_equal(a, b) for a, b in zip(reference, rows, strict=True)):
                raise ValueError(f"検証行が一致しません: {key}")
            ap = average_precision(labels, data["ranked"])
            score = float(ap.mean())
            if not np.isclose(score, records[key].oof_score, rtol=0, atol=1e-12):
                raise ValueError(f"再計算した MAP@7 が記録と一致しません: {key}")
            scores[key], per_row[key] = score, ap
    best = max(scores, key=lambda key: scores[key])
    selected = best
    interval = None
    if best != baseline.run_id:
        assert reference is not None
        interval = group_bootstrap_diff(
            per_row[baseline.run_id],
            per_row[best],
            reference[0],
            transform=lambda ap: ap,
            n_bootstrap=1000,
            seed=SEED,
        )
        if interval.low <= 0:
            selected = baseline.run_id
    selection = {
        "selected_run_id": selected,
        "variant": candidates[selected],
        "data_hash": records[selected].data_hash,
        "split_hash": records[selected].split_hash,
        "evaluation_sha256": records[selected].artifacts[EVALUATION_FILE],
        "highest_cv_run_id": best,
        "cv_map_at_7": scores,
        "bootstrap_best_minus_baseline": asdict(interval) if interval else None,
    }
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
    parser.add_argument("--suffix", default="")
    parser.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        select(args.selection, args.suffix)


if __name__ == "__main__":
    main()
