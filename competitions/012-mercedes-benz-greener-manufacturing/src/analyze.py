"""保存済み run のペア比較と採用モデルの誤差分析。"""

import argparse
import json
from dataclasses import asdict
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from train import DATA, ROOT, SEED, SPLIT_VERSION, categorical_columns, load_data, make_splits

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_regression
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

RUN_VARIANTS = {
    "E00": "baseline",
    "E01": "rare",
    "E02": "strong",
    "E03": "categorical",
    "E04": "interaction",
}
N_BOOTSTRAP = 2000


def audit(run_ids: list[str], selection_path: Path) -> None:
    if len(run_ids) != len(RUN_VARIANTS) or len(set(run_ids)) != len(run_ids):
        raise ValueError("E00〜E04 に対応する重複のない5件の run ID が必要です")
    base_id = run_ids[0]
    x, test, y, sample = load_data()
    groups, splits = make_splits(x)
    roles = split_roles(len(x), splits)
    _, repeated_splits = make_splits(x)
    if not np.array_equal(roles, split_roles(len(x), repeated_splits)):
        raise ValueError("検証分割を再現できません")
    if not np.all((roles == 2).sum(axis=0) == 1):
        raise ValueError("各行をちょうど1回検証する必要があります")
    if any(np.intersect1d(groups[a], groups[b]).size for a, b in splits):
        raise ValueError("同一の車両構成が学習行と検証行をまたいでいます")

    store = RunStore(ROOT)
    split_hash = array_sha256(roles)
    data_hash = files_sha256(
        DATA / name for name in ("train.csv", "test.csv", "sample_submission.csv")
    )
    preds: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    submission_hashes: dict[str, str] = {}
    prediction_artifacts = {"oof.npy", "test.npy", "submission.csv"}
    for run_id, variant in zip(run_ids, RUN_VARIANTS.values(), strict=True):
        record = store.record(run_id)
        if (record.data_hash, record.split_hash) != (data_hash, split_hash):
            raise ValueError(f"{run_id}: データまたは分割のハッシュが一致しません")
        if (
            record.competition_id != ROOT.name
            or record.metric != "r2"
            or record.greater_is_better is not True
            or record.split_version != SPLIT_VERSION
            or record.seed != SEED
        ):
            raise ValueError(f"{run_id}: 評価条件が一致しません")
        if record.params.get("variant") != variant:
            raise ValueError(f"{run_id}: variant は {variant} が必要です")
        if not (prediction_artifacts | {"fold_info.json"}) <= record.artifacts.keys() or any(
            Path(name).name != name for name in record.artifacts
        ):
            raise ValueError(f"{run_id}: 成果物の記録が不正です")
        for name in record.artifacts.keys() - prediction_artifacts:
            store.artifact_path(run_id, name)
        pred = store.load_array(run_id, "oof.npy")
        validate_regression(pred, len(y))
        score = float(r2_score(y, pred))
        if not isclose(score, record.oof_score, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{run_id}: 再計算した OOF R² が記録と一致しません")
        query = store.load_array(run_id, "test.npy")
        validate_regression(query, len(test))
        submission = pd.read_csv(store.artifact_path(run_id, "submission.csv"))
        if list(submission.columns) != ["ID", "y"] or not submission["ID"].equals(sample["ID"]):
            raise ValueError(f"{run_id}: 提出 CSV の列または ID 順が一致しません")
        submitted = submission["y"].to_numpy(dtype=np.float64)
        validate_regression(submitted, len(test))
        if not np.allclose(submitted, query, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{run_id}: 提出 CSV とテスト予測が一致しません")
        preds[run_id] = pred
        scores[run_id] = score
        submission_hashes[run_id] = record.artifacts["submission.csv"]

    base_loss = (preds[base_id] - y) ** 2
    accepted = [base_id]
    comparisons = {}
    for run_id, pred in preds.items():
        if run_id == base_id:
            continue
        diff = group_bootstrap_diff(
            base_loss,
            (pred - y) ** 2,
            groups,
            transform=np.asarray,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )
        comparisons[run_id] = {**asdict(diff), "eligible": diff.high < 0}
        print(f"{run_id} ΔMSE {diff.point:.5f} [{diff.low:.5f}, {diff.high:.5f}]")
        if diff.high < 0:
            accepted.append(run_id)
    winner = max(accepted, key=scores.__getitem__)

    rare = np.zeros(len(y), dtype=bool)
    unseen = np.zeros(len(y), dtype=bool)
    cat = categorical_columns(x)
    for train_idx, valid_idx in splits:
        for name in cat:
            counts = x.iloc[train_idx][name].value_counts()
            frequency = x.iloc[valid_idx][name].map(counts).fillna(0).to_numpy()
            rare[valid_idx] |= (frequency > 0) & (frequency < 5)
            unseen[valid_idx] |= frequency == 0
    tail = y >= np.quantile(y, 0.95)
    ids = pd.read_csv(DATA / "train.csv", usecols=["ID"])["ID"]
    error = preds[winner] - y

    def summarize(name: str, mask: np.ndarray) -> None:
        rows = error[mask]
        if len(rows):
            print(
                f"{name}: n={len(rows)} MAE={np.abs(rows).mean():.4f} "
                f"bias={rows.mean():.4f} SSE割合={(rows**2).sum() / (error**2).sum():.4f}"
            )

    summarize("目的変数の上位5%", tail)
    summarize("希少な既知水準あり・未知水準なし", rare & ~unseen)
    summarize("未知水準あり", unseen)
    summarize("全て5件以上", ~rare & ~unseen)
    for band in pd.qcut(ids, 4).cat.categories:
        mask = ((ids > band.left) & (ids <= band.right)).to_numpy()
        summarize(f"ID {band}", mask)
        print(f"  y平均={y[mask].mean():.4f} X5水準数={x.loc[mask, 'X5'].nunique()}")

    selection = {
        "recorded_after_external_evaluation": run_ids == list(RUN_VARIANTS),
        "baseline_run_id": base_id,
        "run_ids": run_ids,
        "data_hash": data_hash,
        "split_hash": split_hash,
        "split_version": SPLIT_VERSION,
        "bootstrap": {
            "loss_difference": "candidate_mse - baseline_mse",
            "sampling_unit": "vehicle_configuration",
            "confidence_level": 0.95,
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "eligibility": "high < 0",
        },
        "oof_r2": scores,
        "comparisons": comparisons,
        "selected_run_id": winner,
        "submission_sha256": submission_hashes[winner],
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"採用: {winner}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="E00〜E04 に対応する run ID を順に指定")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs" / "selection.json")
    args = parser.parse_args()
    audit(args.run_ids or list(RUN_VARIANTS), args.selection)


if __name__ == "__main__":
    main()
