"""保存済み run を監査し、OOF で選定する。"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dataset import Data, data_hash, load_data
from segmentation import SEED, coverage_bins, duplicate_groups, image_scores, make_folds
from train import ROOT

from kumite.bootstrap import BootstrapDiff, group_bootstrap_diff
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256
from kumite.runs import RunStore


def verify_dataset(root: Path, record: ExperimentRecord) -> tuple[Data, np.ndarray, np.ndarray]:
    if data_hash(root) != record.data_hash:
        raise ValueError("現行データのハッシュが run と一致しません")
    data = load_data(root)
    groups = duplicate_groups(data.train)
    folds = make_folds(data.masks, groups, 5)
    if array_sha256(folds) != record.split_hash:
        raise ValueError("現行データから再計算した分割ハッシュが run と一致しません")
    return data, groups, folds


def verify_split(store: RunStore, run_id: str, groups: np.ndarray, folds: np.ndarray) -> None:
    for name, expected in (("groups.npy", groups), ("folds.npy", folds)):
        actual = store.load_array(run_id, name)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"{run_id}: {name} が再計算結果と一致しません")


def checked_oof(
    oof: np.ndarray, masks: np.ndarray, record: ExperimentRecord, folds: np.ndarray
) -> np.ndarray:
    if (
        oof.shape != masks.shape
        or oof.dtype.kind not in "buif"
        or not ((oof == 0) | (oof == 1)).all()
    ):
        raise ValueError(f"{record.run_id}: OOF は正解と同形状の二値マスクが必要です")
    scores = image_scores(masks, oof.astype(bool))
    fold_scores = [scores[folds == fold].mean() for fold in range(5)]
    if not (
        np.isclose(scores.mean(), record.oof_score, rtol=0, atol=1e-12)
        and len(record.fold_scores) == len(fold_scores)
        and np.allclose(fold_scores, record.fold_scores, rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{record.run_id}: 再採点した OOF が実験ログと一致しません")
    return scores


def sensitivity(
    base: np.ndarray, scores: np.ndarray, groups: np.ndarray
) -> dict[str, tuple[float, BootstrapDiff]]:
    unique, ids, counts = np.unique(groups, return_inverse=True, return_counts=True)
    single = counts[ids] == 1
    conditions = {
        "all_rows": (base, scores, groups),
        "equal_group_weight": (
            np.bincount(ids, weights=base) / counts,
            np.bincount(ids, weights=scores) / counts,
            unique,
        ),
    }
    if single.any():
        conditions["exclude_duplicates"] = (base[single], scores[single], groups[single])
    return {
        name: (
            float(other.mean()),
            group_bootstrap_diff(
                reference,
                other,
                clusters,
                transform=lambda values: values,
                n_bootstrap=10_000,
                seed=SEED,
            ),
        )
        for name, (reference, other, clusters) in conditions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="先頭にベースラインを指定")
    parser.add_argument("--selection", type=Path, help="選定証跡の新規保存先")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("同じ run ID は一度だけ指定してください")
    store = RunStore(ROOT)
    records = [store.record(run_id) for run_id in args.runs]
    baseline = records[0]
    context = (
        "competition_id",
        "metric",
        "greater_is_better",
        "data_hash",
        "split_version",
        "split_hash",
        "seed",
    )
    if any(getattr(record, key) != getattr(baseline, key) for record in records for key in context):
        raise ValueError("評価条件が一致しません")
    data, groups, folds = verify_dataset(ROOT / "data", baseline)
    scores = {}
    for record in records:
        verify_split(store, record.run_id, groups, folds)
        scores[record.run_id] = checked_oof(
            store.load_array(record.run_id, "oof.npy"), data.masks, record, folds
        )
    candidate = max(scores, key=lambda key: float(scores[key].mean()))
    comparisons = {}
    bins = coverage_bins(data.masks)
    for key, values in scores.items():
        by_coverage = [
            f"{values[subset].mean():.6f}" if subset.any() else "該当なし"
            for subset in (bins == 0, bins == 1, bins >= 2)
        ]
        print(f"{key} / 空・小面積・その他 / {' / '.join(by_coverage)}")
        comparisons[key] = sensitivity(scores[baseline.run_id], values, groups)
        for condition, (mean, diff) in comparisons[key].items():
            print(
                f"{key} / {condition} / {mean:.6f} / {diff.point:+.6f} [{diff.low:+.6f}, {diff.high:+.6f}]"
            )
    selected = candidate if comparisons[candidate]["all_rows"][1].low > 0 else baseline.run_id
    submission = store.artifact_path(selected, "submission.csv")
    if args.selection:
        evidence = {
            "created_at": datetime.now(UTC).isoformat(),
            "retrospective": any(record.late_score is not None for record in records),
            "baseline": baseline.run_id,
            "candidate": candidate,
            "selected": selected,
            "data_hash": baseline.data_hash,
            "split_hash": baseline.split_hash,
            "submission_sha256": store.record(selected).artifacts["submission.csv"],
            "bootstrap": {"seed": SEED, "resamples": 10_000, "confidence": 0.95},
            "comparisons": {
                key: {name: {"oof": mean, **asdict(diff)} for name, (mean, diff) in result.items()}
                for key, result in comparisons.items()
            },
        }
        with args.selection.open("x", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(f"採用 {selected} / {submission}")


if __name__ == "__main__":
    main()
