"""CV 記録から最終候補を固定する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import TARGET, TIME, data_files, make_splits
from sklearn.metrics import roc_auc_score

from kumite.checks import validate_multilabel_probabilities
from kumite.crossfit import split_roles
from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SMALL_DELTA = 0.002
COMPLEXITY = {"base": 0, "relative_amount": 1, "clock": 2, "history": 2, "identity": 42}


def selection(store: RunStore, run_ids: list[str]) -> dict[str, object]:
    records = [store.record(run_id) for run_id in run_ids]
    if not records or len(set(run_ids)) != len(run_ids):
        raise ValueError("候補に重複があるか、候補がありません")
    data = store.competition_dir / "data"
    data_hash = files_sha256(data_files(data))
    if any(record.data_hash != data_hash for record in records):
        raise ValueError("候補 run のデータハッシュが現在の入力と一致しません")
    train = pd.read_csv(data / "train_transaction.csv", usecols=[TIME, TARGET])
    test = pd.read_csv(data / "test_transaction.csv", usecols=[TIME])
    splits, split_version = make_splits(train, test)
    expected_roles = split_roles(len(train), splits)
    split_hash = array_sha256(expected_roles)
    y = train[TARGET].to_numpy()
    validated = (expected_roles == 2).any(axis=0)
    common_params = {k: v for k, v in records[0].params.items() if k not in {"trial", "seed"}}
    comparisons: dict[tuple[int, str], ExperimentRecord] = {}
    scores: dict[str, float] = {}
    for record in records:
        trial = str(record.params.get("trial"))
        run_id = record.run_id
        if (
            (record.split_hash, record.split_version) != (split_hash, split_version)
            or record.competition_id != "7"
            or record.metric != "ROC AUC"
            or not record.greater_is_better
            or record.model != "LightGBM"
            or trial not in COMPLEXITY
            or record.seed not in {42, 43}
            or record.params.get("seed") != record.seed
            or record.params != {**common_params, "trial": trial, "seed": record.seed}
        ):
            raise ValueError(f"{run_id} は比較可能な CV run ではありません")
        roles = store.load_array(run_id, "split_roles.npy")
        if array_sha256(roles) != split_hash:
            raise ValueError(f"{run_id} の fold 成果物が不正です")
        oof = store.load_array(run_id, "oof.npy")
        if (
            oof.shape != y.shape
            or not np.issubdtype(oof.dtype, np.floating)
            or not np.isnan(oof[~validated]).all()
        ):
            raise ValueError(f"{run_id} の OOF 対象行が一致しません")
        validate_multilabel_probabilities(oof[validated, None], int(validated.sum()), 1)
        fold_scores = [float(roc_auc_score(y[va], oof[va])) for _, va in splits]
        oof_score = float(roc_auc_score(y[validated], oof[validated]))
        if fold_scores != record.fold_scores or oof_score != record.oof_score:
            raise ValueError(f"{run_id} の保存済み AUC が OOF の再計算と一致しません")
        scores[run_id] = float(np.mean(fold_scores))
        key = (record.seed, trial)
        if key in comparisons:
            raise ValueError("同じ seed と特徴設定の候補が複数あります")
        comparisons[key] = record

    def score(record: ExperimentRecord) -> float:
        return scores[record.run_id]

    if (42, "base") not in comparisons:
        raise ValueError("seed 42 の基準 run がありません")
    base = comparisons[42, "base"]
    eligible = [base]
    excluded = {}
    for (seed, trial), record in comparisons.items():
        if seed != 42 or trial == "base":
            continue
        delta = score(record) - score(base)
        if delta <= 0:
            excluded[record.run_id] = "基準より改善していない"
            continue
        if delta <= SMALL_DELTA:
            check_base = comparisons.get((43, "base"))
            check_trial = comparisons.get((43, trial))
            if check_base is None or check_trial is None:
                raise ValueError(f"{record.run_id} は seed 43 のペアが必要です")
            if score(check_trial) <= score(check_base):
                excluded[record.run_id] = "seed 43 で改善しない"
                continue
        eligible.append(record)

    winner = min(
        eligible,
        key=lambda record: (
            -score(record),
            COMPLEXITY[str(record.params["trial"])],
            record.run_id,
        ),
    )
    return {
        "selected": winner.run_id,
        "data_hash": data_hash,
        "split_hash": split_hash,
        "split_version": split_version,
        "small_delta": SMALL_DELTA,
        "excluded": excluded,
        "candidates": {
            record.run_id: {
                "mean_auc": score(record),
                "seed": record.seed,
                "oof_sha256": record.artifacts["oof.npy"],
            }
            for record in records
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    result = selection(RunStore(ROOT), args.runs)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(result["selected"])


if __name__ == "__main__":
    main()
