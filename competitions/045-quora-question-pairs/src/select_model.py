"""採点済みの OOF から構成を選び、照合済みの提出成果物を確定する。"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from inputs import read_verified_train
from predict import ROOT
from train import score

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore

CANDIDATES = {"E01": "E01", "E02-cv": "E02", "E03": "E03", "E04": "E04", "E05": "E05"}


def comparison_key(record: ExperimentRecord) -> tuple[str, str, str, int]:
    return record.data_hash, record.split_version, record.split_hash, record.seed


def verified_score(
    store: RunStore, record: ExperimentRecord, experiment: str, frame: pd.DataFrame
) -> float:
    if (
        record.metric != "log_loss"
        or record.greater_is_better is not False
        or record.params.get("experiment") != experiment
    ):
        raise ValueError(f"{record.run_id} の指標または実験定義が一致しません")
    with np.load(store.artifact_path(record.run_id, "split.npz")) as saved:
        roles = saved["roles"]
        if not np.array_equal(saved["ids"], frame["id"].to_numpy()):
            raise ValueError("分割と学習データの行 ID が一致しません")
    if (
        roles.shape != (5, len(frame))
        or roles.dtype != np.int8
        or not np.isin(roles, [0, 1, 2]).all()
        or ((roles == 2).sum(axis=0) > 1).any()
        or hashlib.sha256(roles.tobytes()).hexdigest() != record.split_hash
    ):
        raise ValueError(f"{record.run_id} の分割が不正です")
    oof = store.load_array(record.run_id, "oof.npy")
    valid = (roles == 2).any(axis=0)
    if (
        oof.shape != (len(frame),)
        or not valid.any()
        or not np.isfinite(oof[valid]).all()
        or not np.isnan(oof[~valid]).all()
        or ((oof[valid] < 0) | (oof[valid] > 1)).any()
    ):
        raise ValueError(f"{record.run_id} の OOF が不正です")
    loss = score(frame["is_duplicate"].to_numpy()[valid], oof[valid])
    if not np.isclose(loss, record.oof_score, rtol=0, atol=1e-12):
        raise ValueError(f"{record.run_id} の OOF スコアが記録と一致しません")
    return loss


def submission_record(
    store: RunStore, winner: ExperimentRecord, submission_run: str | None, frame: pd.DataFrame
) -> ExperimentRecord:
    if submission_run is None and "submission.csv" not in winner.artifacts:
        raise ValueError(
            f"CV 勝者は {winner.run_id} ですが提出成果物が未生成です。"
            "同じ構成・seed で別 run ID の train.py を実行し、--submission-run に指定してください"
        )
    if submission_run is None:
        return winner
    record = store.record(submission_run)
    if record.run_id == winner.run_id:
        return winner
    trial_params = {key: value for key, value in record.params.items() if key != "cv_only"}
    winning_params = {key: value for key, value in winner.params.items() if key != "cv_only"}
    if (
        comparison_key(record) != comparison_key(winner)
        or record.model != winner.model
        or record.data_version != winner.data_version
        or trial_params != winning_params
        or record.artifacts.get("oof.npy") != winner.artifacts["oof.npy"]
    ):
        raise ValueError("提出 run が CV 勝者の構成・分割・OOF と一致しません")
    verified_score(store, record, CANDIDATES[winner.run_id], frame)
    return record


def verify_submission(store: RunStore, record: ExperimentRecord, root: Path) -> None:
    csv = pd.read_csv(
        store.artifact_path(record.run_id, "submission.csv"), dtype={"is_duplicate": str}
    )
    ids = pd.read_csv(root / "data" / "sample_submission.csv")["test_id"]
    test = store.load_array(record.run_id, "test.npy")
    if (
        list(csv.columns) != ["test_id", "is_duplicate"]
        or ids.isna().any()
        or ids.duplicated().any()
        or not csv["test_id"].equals(ids)
        or test.shape != (len(ids),)
        or not np.isfinite(test).all()
        or ((test < 0) | (test > 1)).any()
        or not np.array_equal(csv["is_duplicate"].map(float).to_numpy(), test)
    ):
        raise ValueError("提出 CSV が提出見本または保存済みテスト確率と一致しません")


def select(root: Path, submission_run: str | None) -> dict[str, Any]:
    store = RunStore(root)
    log_path = root / "experiments.jsonl"
    log_hash = hashlib.sha256(log_path.read_bytes()).hexdigest()
    candidates = [store.record(run_id) for run_id in CANDIDATES]
    if len({comparison_key(item) for item in candidates}) != 1 or candidates[0].seed != 42:
        raise ValueError("seed 42 の候補が同じデータ・分割を使用していません")
    frame = read_verified_train(root, candidates[0].data_hash)
    scores = {
        item.run_id: verified_score(store, item, CANDIDATES[item.run_id], frame)
        for item in candidates
    }
    first, second, *_ = sorted(candidates, key=lambda item: (scores[item.run_id], item.run_id))
    winner = first
    means = {}
    repeats = []
    if scores[second.run_id] - scores[first.run_id] < 0.001:
        repeats = [store.record(f"{CANDIDATES[item.run_id]}-seed43") for item in (first, second)]
        if (
            comparison_key(repeats[0]) != comparison_key(repeats[1])
            or repeats[0].seed != 43
            or repeats[0].data_hash != first.data_hash
        ):
            raise ValueError("seed 43 の再測定条件が一致しません")
        means = {
            item.run_id: (
                scores[item.run_id] + verified_score(store, repeat, CANDIDATES[item.run_id], frame)
            )
            / 2
            for item, repeat in zip((first, second), repeats, strict=True)
        }
        winner = min((first, second), key=lambda item: (means[item.run_id], item.run_id))
    result = submission_record(store, winner, submission_run, frame)
    verify_submission(store, result, root)
    if hashlib.sha256(log_path.read_bytes()).hexdigest() != log_hash:
        raise ValueError("照合中に実験ログが更新されました")
    return {
        "selected_run": result.run_id,
        "selection_evidence_run": winner.run_id,
        "experiment": CANDIDATES[winner.run_id],
        "reason": "2 seed の平均 Log Loss が最小" if means else "seed 42 の OOF Log Loss が最小",
        "seed42": scores,
        "two_seed_mean": means,
        "candidates": {
            item.run_id: {"oof_sha256": item.artifacts["oof.npy"], "split_hash": item.split_hash}
            for item in [*candidates, *repeats]
        },
        "data_hash": result.data_hash,
        "submission_sha256": result.artifacts["submission.csv"],
        "experiments_sha256": log_hash,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "retrospective": result.late_score is not None or winner.late_score is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-run")
    args = parser.parse_args()
    try:
        selection = select(ROOT, args.submission_run)
        with (ROOT / "selection.json").open("x", encoding="utf-8") as out:
            json.dump(selection, out, ensure_ascii=False, indent=2)
            out.write("\n")
    except (LookupError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
