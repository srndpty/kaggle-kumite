"""同じ OOF 行のペア比較で採用 run を固定し、誤差を集計する。"""

import argparse
import json
from datetime import UTC, datetime
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import GROUP, ID, TARGET
from model import LABELS, SEED, qwk
from train import ROOT, RUN_VARIANTS, prepare

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore

KEYS = [ID, GROUP, TARGET, "PhotoAmt", "group", "fold"]
WEIGHTS = (LABELS[:, None] - LABELS) ** 2


def verified_artifact(store: RunStore, record: ExperimentRecord, name: str) -> Path:
    path = store.runs_dir / record.run_id / name
    if file_sha256(path) != record.artifacts.get(name):
        raise ValueError(f"{record.run_id}/{name} のハッシュが記録と一致しません")
    return path


def read_oof(store: RunStore, record: ExperimentRecord) -> pd.DataFrame:
    return pd.read_csv(
        verified_artifact(store, record, "oof.csv"),
        dtype={ID: str, GROUP: str, "group": str},
        converters={"score": float},
    )


def verify_oof(
    record: ExperimentRecord, frame: pd.DataFrame, official: pd.DataFrame, data_hash: str
) -> float:
    if record.run_id not in RUN_VARIANTS or (
        record.competition_id,
        record.data_version,
        record.model,
        record.metric,
        record.greater_is_better,
        record.seed,
        record.split_version,
    ) != ("016", RUN_VARIANTS[record.run_id], "CatBoost", "QWK", True, SEED, "rescuer-image-v1"):
        raise ValueError(f"{record.run_id} の実験契約が無効です")
    if record.data_hash != data_hash or record.split_hash != array_sha256(
        official["fold"].to_numpy()
    ):
        raise ValueError(f"{record.run_id} のデータまたは分割のハッシュが一致しません")
    if not frame[ID].is_unique or not frame[KEYS].equals(official[KEYS]):
        raise ValueError(f"{record.run_id} の OOF 行が公式データの割当と一致しません")
    pred = frame["prediction"]
    if not pd.api.types.is_integer_dtype(pred) or not pred.isin(LABELS).all():
        raise ValueError(f"{record.run_id} の OOF 予測は 0〜4 の整数である必要があります")
    score = qwk(frame[TARGET].to_numpy(), pred.to_numpy())
    if not isclose(score, record.oof_score, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"{record.run_id} の OOF QWK が記録と一致しません")
    return score


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


def error_summary(frame: pd.DataFrame) -> dict:
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
        "image": summarize(pd.Series(np.where(frame["readable_images"] > 0, "あり", "なし"))),
        "rescuer_listings": summarize(
            pd.cut(listing_counts, [0, 1, 5, 20, np.inf], labels=["1", "2–5", "6–20", "21+"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    store = RunStore(ROOT)
    records = {record.run_id: record for record in store.records()}

    candidates = sorted(set(args.candidates))
    frames = {run_id: read_oof(store, records[run_id]) for run_id in {"E01", *candidates}}
    official, _, _, _, _, data_hash, _ = prepare(args.data, "baseline", audit=True)
    baseline = frames["E01"]
    baseline_record = records["E01"]
    best_id, best_score = "E01", verify_oof(baseline_record, baseline, official, data_hash)
    comparisons = {}
    for run_id in candidates:
        record = records[run_id]
        frame = frames[run_id]
        score = verify_oof(record, frame, official, data_hash)
        pred = frame["prediction"].to_numpy()
        interval = paired_interval(baseline, pred)
        comparisons[run_id] = {
            "qwk": score,
            "delta": score - baseline_record.oof_score,
            "interval": interval,
        }
        if interval[0] > 0 and (best_id == "E01" or score > best_score):
            best_id, best_score = run_id, score

    summary = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "data_hash": baseline_record.data_hash,
        "split_hash": baseline_record.split_hash,
        "baseline_qwk": baseline_record.oof_score,
        "selected": best_id,
        "qwk": best_score,
        "comparisons": comparisons,
        "errors": error_summary(
            frames[best_id].assign(readable_images=official["readable_images"])
        ),
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
