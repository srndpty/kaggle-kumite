"""固定 CV の実験記録と OOF を照合し、選定証跡を保存する。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from train import ROOT, SEED, SPLIT_VERSION, data_hash, load, make_model, make_splits, score

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_probabilities
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256
from kumite.runs import RunStore

CANDIDATES = {
    "E00": "E00",
    "E01": "E01",
    "E02": "E02",
    "E03": "E03",
    "E04": "E04",
    "E03-r2": "E03",
}
BOOTSTRAP_DRAWS = 2000


def expected_params(run_id: str) -> dict:
    params = {"variant": CANDIDATES[run_id], **make_model().get_params()}
    if run_id in {"E00", "E01"}:
        # 初期 2 run はコンストラクタに明示した設定だけを記録した。
        keys = (
            "variant",
            "objective",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "n_jobs",
            "random_state",
            "deterministic",
            "force_col_wise",
            "verbosity",
        )
        return {key: params[key] for key in keys}
    if run_id == "E03-r2":
        params["missing_category_policy"] = "always-reserve"
    return params


def verified_losses(
    store: RunStore,
    run_id: str,
    y: np.ndarray,
    current_data_hash: str,
    roles: np.ndarray,
    split_hash: str,
) -> tuple[np.ndarray, str]:
    record = store.record(run_id)
    if record.data_hash != current_data_hash:
        raise ValueError(f"{run_id} のデータハッシュが一致しません")
    if (
        record.competition_id,
        record.metric,
        record.greater_is_better,
        record.data_version,
        record.model,
        record.seed,
        record.split_version,
        record.params,
    ) != (
        "010",
        "LogLoss",
        False,
        "kaggle-v1",
        "LightGBM",
        SEED,
        SPLIT_VERSION,
        expected_params(run_id),
    ):
        raise ValueError(f"{run_id} の実験設定が一致しません")
    if record.split_hash != split_hash or not np.array_equal(
        roles, store.load_array(run_id, "split_roles.npy")
    ):
        raise ValueError(f"{run_id} の fold が一致しません")

    probability = store.load_array(run_id, "oof.npy")
    if probability.shape != y.shape or probability.dtype.kind != "f":
        raise ValueError(f"{run_id} の OOF の形状または型が不正です")
    validate_probabilities(np.column_stack((1 - probability, probability)), len(y), 2)
    if not np.isclose(score(y, probability), record.oof_score, rtol=0, atol=1e-12):
        raise ValueError(f"{run_id} の OOF スコアが実験ログと一致しません")
    actual_class_probability = np.where(y == 1, probability, 1 - probability)
    return -np.log(
        np.clip(actual_class_probability, np.finfo(probability.dtype).eps, 1)
    ), record.artifacts["oof.npy"]


def select(data: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    x, y, _, _ = load(data)
    current_data_hash = data_hash(data)
    roles = split_roles(len(y), make_splits(x, y))
    split_hash = array_sha256(roles)
    store = RunStore(ROOT)
    verified = {
        run_id: verified_losses(store, run_id, y, current_data_hash, roles, split_hash)
        for run_id in CANDIDATES
    }
    baseline = verified["E00"][0]
    groups = np.arange(len(y))
    missing = x.isna().any(axis=1).to_numpy()
    cat_missing = x.select_dtypes(exclude="number").isna().any(axis=1).to_numpy()
    results = {}
    selection = "E00"
    best = 0.0
    for run_id, (loss, oof_hash) in verified.items():
        diff = group_bootstrap_diff(
            baseline, loss, groups, transform=np.asarray, n_bootstrap=BOOTSTRAP_DRAWS, seed=SEED
        )
        if diff.high < 0 and diff.point < best:
            best = diff.point
            selection = run_id
        results[run_id] = {
            "oof_sha256": oof_hash,
            "log_loss": float(loss.mean()),
            "delta": diff.point,
            "ci95": [diff.low, diff.high],
            "with_missing": float(loss[missing].mean()),
            "without_missing": float(loss[~missing].mean()),
            "with_categorical_missing": float(loss[cat_missing].mean()),
        }
    report = {
        "selected": selection,
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_hash": current_data_hash,
        "split_hash": split_hash,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
        "runs": results,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    print(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "selection-review.json")
    args = parser.parse_args()
    select(args.data, args.output)


if __name__ == "__main__":
    main()
