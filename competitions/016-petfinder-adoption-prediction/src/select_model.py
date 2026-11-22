"""同じ OOF 行のペア比較で採用 run を固定し、誤差を集計する。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import GROUP, ID, TARGET
from model import LABELS, SEED, qwk
from train import ROOT

from kumite.runs import RunStore

KEYS = [ID, GROUP, TARGET, "PhotoAmt", "group", "fold"]
WEIGHTS = (LABELS[:, None] - LABELS) ** 2


def group_tables(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(groups * 25 + y * 5 + pred, minlength=count * 25).reshape(count, 5, 5)


def paired_interval(frame: pd.DataFrame, pred: np.ndarray) -> list[float]:
    y = frame[TARGET].to_numpy()
    groups, unique = pd.factorize(frame["group"])
    count = len(unique)
    base = group_tables(y, frame["prediction"].to_numpy(), groups, count)
    candidate = group_tables(y, pred, groups, count)
    rng = np.random.default_rng(SEED)

    def score(table: np.ndarray) -> float:
        expected = table.sum(axis=1) @ WEIGHTS @ table.sum(axis=0)
        if expected == 0:
            return float("nan")
        return float(1 - (table * WEIGHTS).sum() * table.sum() / expected)

    differences = []
    for _ in range(2000):
        draw = rng.integers(0, count, size=count)
        differences.append(score(candidate[draw].sum(axis=0)) - score(base[draw].sum(axis=0)))
    valid = np.asarray(differences)
    if not np.isfinite(valid).all():
        raise ValueError("再標本の QWK を計算できません")
    return np.percentile(valid, [2.5, 97.5]).tolist()


def error_summary(frame: pd.DataFrame, features: pd.DataFrame) -> dict:
    pet_ids = frame[ID]
    image_counts = features.set_index(ID)["readable_images"].reindex(pet_ids)
    if image_counts.isna().any():
        raise ValueError("画像件数と OOF の PetID が一致しません")
    listing_counts = frame.groupby(GROUP)[ID].transform("size")
    error = (frame[TARGET] - frame["prediction"]).abs()

    def summarize(labels: pd.Series) -> dict:
        table = pd.DataFrame({"error": error, "large_error": error >= 2, "slice": labels})
        return (
            table.groupby("slice", observed=True)
            .agg(
                rows=("error", "size"),
                mae=("error", "mean"),
                large_error_rate=("large_error", "mean"),
            )
            .to_dict(orient="index")
        )

    return {
        "class": summarize(frame[TARGET]),
        "image": summarize(pd.Series(np.where(image_counts.to_numpy() > 0, "あり", "なし"))),
        "rescuer_listings": summarize(
            pd.cut(listing_counts, [0, 1, 5, 20, np.inf], labels=["1", "2–5", "6–20", "21+"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = RunStore(ROOT)
    records = {record.run_id: record for record in store.records()}

    def load(run_id: str) -> pd.DataFrame:
        return pd.read_csv(
            store.runs_dir / run_id / "oof.csv", dtype={ID: str, GROUP: str, "group": str}
        )

    baseline = load("E01")
    baseline_record = records["E01"]
    best_id, best_score = "E01", qwk(baseline[TARGET].to_numpy(), baseline["prediction"].to_numpy())
    comparisons = {}
    for run_id in sorted(set(args.candidates)):
        record = records[run_id]
        frame = load(run_id)
        if (record.data_hash, record.split_hash) != (
            baseline_record.data_hash,
            baseline_record.split_hash,
        ) or not frame[KEYS].equals(baseline[KEYS]):
            raise ValueError(f"{run_id} のデータまたは分割が E01 と異なります")
        pred = frame["prediction"].to_numpy()
        interval = paired_interval(baseline, pred)
        score = qwk(frame[TARGET].to_numpy(), pred)
        comparisons[run_id] = {
            "qwk": score,
            "delta": score - baseline_record.oof_score,
            "interval": interval,
        }
        if interval[0] > 0 and (best_id == "E01" or score > best_score):
            best_id, best_score = run_id, score

    features = pd.read_csv(ROOT / "outputs" / "eda_images.csv", dtype={ID: str})
    summary = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "data_hash": baseline_record.data_hash,
        "split_hash": baseline_record.split_hash,
        "baseline_qwk": baseline_record.oof_score,
        "selected": best_id,
        "qwk": best_score,
        "comparisons": comparisons,
        "errors": error_summary(load(best_id), features),
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
