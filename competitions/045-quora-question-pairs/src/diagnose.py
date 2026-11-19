"""同一の検証ペアで、質問共有を許した学習集合と主分割を比較する。"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inputs import read_verified_train
from predict import ROOT
from split import node_splits, question_nodes
from train import fit_model, pair_text, score

from kumite.crossfit import split_roles
from kumite.experiments import find_record, read_records
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version
from kumite.runs import RunStore


def fold_evidence(
    nodes: np.ndarray, y: np.ndarray, control: np.ndarray, shared: np.ndarray
) -> dict[str, Any]:
    valid = np.flatnonzero(control == 2)
    if not np.array_equal(valid, np.flatnonzero(shared == 2)):
        raise ValueError("診断の検証ペアが一致しません")
    control_train = control == 1
    shared_train = shared == 1
    control_count = int(control_train.sum()), int(y[control_train].sum())
    shared_count = int(shared_train.sum()), int(y[shared_train].sum())
    if control_count != shared_count:
        raise ValueError("診断の学習行数または陽性数が一致しません")
    valid_nodes = nodes[valid]
    if np.intersect1d(valid_nodes, nodes[control_train]).size:
        raise ValueError("対照条件で検証と学習が質問を共有しています")
    matched = np.intersect1d(valid_nodes, nodes[shared_train])
    return {
        "control_train": dict(zip(("rows", "positive"), control_count, strict=True)),
        "shared_train": dict(zip(("rows", "positive"), shared_count, strict=True)),
        "shared_question_nodes": len(matched),
        "shared_validation_pairs": int(np.isin(valid_nodes, matched).any(axis=1).sum()),
        "overlapping_train_rows": int((control_train & shared_train).sum()),
        "different_train_rows": int((control_train ^ shared_train).sum()),
        "validation_pairs": len(valid),
        "validation_pairs_identical": True,
    }


def diagnose(by_node: bool) -> tuple[list[dict[str, Any]], float, np.ndarray, np.ndarray]:
    store = RunStore(ROOT)
    frame = read_verified_train(ROOT, store.record("E01").data_hash)
    pairs = pair_text(frame)
    nodes = question_nodes(frame)
    y = frame["is_duplicate"].to_numpy(dtype=np.int64)
    oof = store.load_array("E01", "oof.npy")
    with np.load(store.artifact_path("E01", "split.npz")) as saved:
        if not np.array_equal(saved["ids"], frame["id"].to_numpy()):
            raise ValueError("学習データの行 ID が E01 と一致しません")
        splits = [
            (np.flatnonzero(roles == 1), np.flatnonzero(roles == 2)) for roles in saved["roles"]
        ]
    if by_node:
        splits = node_splits(nodes, 42)
    control_roles = split_roles(len(y), splits)
    ordered = np.sort(nodes, axis=1).astype(np.int64)
    keys = ordered[:, 0] * (int(nodes.max()) + 1) + ordered[:, 1]
    measures = []
    losses = []
    weights = []
    sampled_roles = np.zeros((len(splits), len(y)), dtype=np.int8)
    for fold, (train, valid) in enumerate(splits):
        eligible = ~np.isin(keys, keys[valid])
        rng = np.random.default_rng(42 + fold)
        sampled = []
        shortages = {}
        for target in (0, 1):
            count = int((y[train] == target).sum())
            candidates = np.flatnonzero(eligible & (y == target))
            shortages[target] = max(count - len(candidates), 0)
            sampled.append(rng.choice(candidates, min(count, len(candidates)), replace=False))
        if any(shortages.values()):
            measures.append({"fold": fold, "shortages": shortages})
            continue
        rows = np.sort(np.concatenate(sampled))
        sampled_roles[fold, rows] = 1
        sampled_roles[fold, valid] = 2
        evidence = fold_evidence(nodes, y, control_roles[fold], sampled_roles[fold])
        features, model = fit_model(pairs[rows], y[rows], "E01", 42)
        pred = model.predict_proba(features.transform(pairs[valid]))[:, 1]
        control = oof[valid]
        if by_node:
            features, model = fit_model(pairs[train], y[train], "E01", 42)
            control = model.predict_proba(features.transform(pairs[valid]))[:, 1]
        shared = np.isin(nodes[valid], nodes[rows]).sum(axis=1)
        groups = {}
        for count in (0, 1, 2):
            mask = shared == count
            if mask.any():
                groups[count] = {
                    "rows": int(mask.sum()),
                    "control": score(y[valid][mask], control[mask]),
                    "shared": score(y[valid][mask], pred[mask]),
                }
        loss = score(y[valid], pred)
        losses.append(loss)
        weights.append(len(valid))
        measures.append(
            {
                "fold": fold,
                "control": score(y[valid], control),
                "shared": loss,
                "groups": groups,
                "comparison": evidence,
            }
        )
    if not weights:
        raise ValueError("件数を揃えた比較が成立した fold がありません")
    return measures, float(np.average(losses, weights=weights)), sampled_roles, control_roles


def audit_saved_run(store: RunStore, run_id: str) -> dict[str, Any]:
    frame = read_verified_train(ROOT, store.record(run_id).data_hash)
    y = frame["is_duplicate"].to_numpy(dtype=np.int64)
    nodes = question_nodes(frame)
    shared = store.load_array(run_id, "roles.npy")
    control = store.load_array(run_id, "control_roles.npy")
    original = json.loads(store.artifact_path(run_id, "diagnosis.json").read_text(encoding="utf-8"))
    if shared.shape != control.shape or shared.shape != (len(original["folds"]), len(y)):
        raise ValueError("保存済み診断の形状が一致しません")
    for fold, (a, b) in enumerate(zip(control, shared, strict=True)):
        entry = original["folds"][fold]
        if entry["fold"] != fold:
            raise ValueError("診断記録の fold が一致しません")
        if "shortages" not in entry:
            entry["comparison"] = fold_evidence(nodes, y, a, b)
    return {
        "run_id": run_id,
        "inputs": store.input_hashes(run_id, "roles.npy", "control_roles.npy", "diagnosis.json"),
        "folds": original["folds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-holdout", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--audit-run", help="保存済み分割から比較証跡を再構成する")
    args = parser.parse_args()
    by_node = args.node_holdout
    store = RunStore(ROOT)
    if args.audit_run:
        print(json.dumps(audit_saved_run(store, args.audit_run), indent=2))
        return
    run_id = args.run_id or ("E01-shared-r2" if by_node else "E01-shared")
    with store.reserve(run_id) as reservation:
        measures, loss, roles, control_roles = diagnose(by_node)
        baseline = find_record(read_records(ROOT / "experiments.jsonl"), "E01")
        hashes = store.input_hashes("E01", "oof.npy", "split.npz")
        record = replace(
            baseline,
            run_id=reservation.run_id,
            split_version="node-holdout-shared-seed42-v1"
            if by_node
            else "matched-shared-seed42-v1",
            split_hash=hashlib.sha256(roles.tobytes()).hexdigest(),
            hypothesis="学習側と質問を共有できる条件では損失が変わる",
            changed_one_thing="学習件数と陽性件数を揃えて質問共有を許す",
            fold_scores=[fold["shared"] for fold in measures if "shared" in fold],
            oof_score=loss,
            late_score=None,
            runtime_sec=None,
            peak_memory_mb=None,
            code_version=code_version(REPO_ROOT),
            inputs=hashes,
            artifacts={},
            conclusion="E01 と同一検証ペアで比較する分割診断。",
        )
        summary = json.dumps({"inputs": hashes, "folds": measures}, indent=2)

        def write(path: Path) -> None:
            (path / "diagnosis.json").write_text(summary, encoding="utf-8")
            np.save(path / "roles.npy", roles)
            np.save(path / "control_roles.npy", control_roles)

        store.publish(reservation, record, write)
        print(summary)


if __name__ == "__main__":
    main()
