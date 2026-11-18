"""保存済み分割を再生成し、事前の採用規則だけで最終提出を固定する。"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from dataset import CLASSES, OUTPUT, ROOT, load_prepared, make_folds, submission
from supervision import FINE_CLASSES

from kumite.experiments import ExperimentRecord, compare_runs
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore

SEED42_RUNS = {"E01": "E01", "E02": "E02", "E03": "E03", "E04": "E04", "E05-r2": "E05"}
RUN_DEFINITIONS = {
    **{run: (experiment, 42) for run, experiment in SEED42_RUNS.items()},
    "E01-s43": ("E01", 43),
    "E05-s43": ("E05", 43),
}


def validate_run_definition(record: ExperimentRecord) -> None:
    """この実験系列で許可したIDと宣言の対応を検査する。"""
    expected = RUN_DEFINITIONS.get(record.run_id)
    if expected is None:
        raise ValueError(f"許可していない選定run ID: {record.run_id}")
    experiment, seed = expected
    expected_classes = list(FINE_CLASSES if experiment == "E05" else CLASSES)
    params = record.params
    if (
        params.get("experiment") != experiment
        or record.seed != seed
        or record.model != "SmallCNN"
        or record.metric != "accuracy"
        or not record.greater_is_better
        or record.data_version != "audio-v1"
        or params.get("benchmark") is not False
        or params.get("random_split") is not False
        or params.get("failed")
        or params.get("classes") != list(CLASSES)
    ):
        raise ValueError(f"run IDと実験定義が一致しません: {record.run_id}")
    # 初期E01〜E04にはtraining_classesがない。12出力の実体はcheckpointでも検証する。
    if "training_classes" in params:
        if params["training_classes"] != expected_classes:
            raise ValueError(f"学習クラスが実験定義と一致しません: {record.run_id}")
    elif record.run_id not in {"E01", "E02", "E03", "E04"}:
        raise ValueError(f"学習クラスの記録がありません: {record.run_id}")
    if len(record.fold_scores) != 3 or not np.isfinite(record.fold_scores).all():
        raise ValueError(f"3foldの計測値がありません: {record.run_id}")


def validate_run_artifacts(store: RunStore, record: ExperimentRecord) -> None:
    validate_run_definition(record)
    config = json.loads(
        store.artifact_path(record.run_id, "config.json").read_text(encoding="utf-8")
    )
    if config != record.params["config"]:
        raise ValueError(f"保存した学習設定が一致しません: {record.run_id}")
    for k in range(3):
        checkpoint = torch.load(
            store.artifact_path(record.run_id, f"fold{k}.pt"), map_location="cpu", weights_only=True
        )
        expected_outputs = 31 if record.params["experiment"] == "E05" else 12
        if (
            checkpoint["state_dict"]["layers.14.weight"].shape != (expected_outputs, 128)
            or checkpoint["state_dict"]["layers.14.bias"].shape != (expected_outputs,)
            or checkpoint["linear"] is not (record.params["experiment"] == "E02")
        ):
            raise ValueError(f"checkpointが実験定義と一致しません: {record.run_id}/fold{k}")


def choose_candidate(records: list[ExperimentRecord]) -> ExperimentRecord:
    eligible = [
        r
        for r in records
        if r.seed == 42
        and not r.params.get("failed")
        and not r.params.get("benchmark")
        and not r.params.get("random_split")
        and r.params.get("experiment") in {f"E{k:02}" for k in range(1, 6)}
    ]
    if {r.params["experiment"] for r in eligible} != {f"E{k:02}" for k in range(1, 6)}:
        raise ValueError("E01〜E05が揃っていません")
    if len(eligible) != 5:
        raise ValueError("候補の重複があります。恣意的な再試行選択を防ぐため停止します")
    for record in eligible:
        validate_run_definition(record)
    return max(eligible, key=lambda r: (float(np.mean(r.fold_scores)), -(r.runtime_sec or 0)))


def freeze_candidate() -> None:
    store = RunStore(ROOT)
    candidate = choose_candidate(store.records())
    p5 = json.loads((ROOT / "p5_registration.json").read_text(encoding="utf-8"))
    parent = p5["parent"] if candidate.params["experiment"] == "E05" else "E01"
    value = dict(
        candidate=candidate.run_id,
        parent=parent,
        data_hash=candidate.data_hash,
        split_hash=candidate.split_hash,
        fold_scores=candidate.fold_scores,
        artifacts=candidate.artifacts,
    )
    with (OUTPUT / "candidate.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(f"確認seed前の候補固定: {candidate.run_id} / 比較元 {parent}")


def build_selection() -> dict:
    store = RunStore(ROOT)
    records = store.records()
    candidate = choose_candidate(records)
    frozen = json.loads((OUTPUT / "candidate.json").read_text(encoding="utf-8"))
    if frozen["candidate"] != candidate.run_id or frozen["artifacts"] != candidate.artifacts:
        raise ValueError("確認seed前に固定した候補と一致しません")
    frame, test, _, _, summary = load_prepared(deep=True)
    p5 = json.loads((ROOT / "p5_registration.json").read_text(encoding="utf-8"))
    parent_experiment = p5["parent"] if candidate.params["experiment"] == "E05" else "E01"
    if parent_experiment != "E01":
        raise ValueError("事前登録の比較元が実験定義と一致しません")
    for name, expected in {
        "parent": parent_experiment,
        "data_hash": candidate.data_hash,
        "split_hash": candidate.split_hash,
        "fold_scores": candidate.fold_scores,
    }.items():
        if frozen[name] != expected:
            raise ValueError(f"固定した候補の{name}が一致しません")
    parent = next(r for r in records if r.run_id == parent_experiment)
    relevant = [parent, candidate]
    if candidate.run_id != parent.run_id:
        relevant += [store.record(f"{r.params['experiment']}-s43") for r in (parent, candidate)]
    checked = {r.run_id: r for r in [*(store.record(run) for run in SEED42_RUNS), *relevant]}
    expected_config = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    for record in checked.values():
        validate_run_artifacts(store, record)
        if record.data_hash != summary["data_hash"] or record.params["config"] != expected_config:
            raise ValueError("データ・学習設定が一致しません")
        folds = make_folds(frame, record.seed)
        if (
            record.split_hash != array_sha256(folds)
            or record.split_version != f"speaker-background-3fold-v1-seed{record.seed}"
        ):
            raise ValueError("分割を再生成できません")
        saved = pd.read_csv(store.artifact_path(record.run_id, "manifest.csv"))
        if saved.id.tolist() != frame.id.tolist() or not np.array_equal(saved.fold, folds):
            raise ValueError("保存された行順・foldが不正です")
        for name in record.artifacts:
            store.artifact_path(record.run_id, name)
    adopted = parent
    differences: dict[str, list[float]] = {}
    if candidate.run_id != parent.run_id:
        first = compare_runs(parent, candidate)
        second = compare_runs(relevant[2], relevant[3])
        differences = {"42": first.diffs, "43": second.diffs}
        if first.mean_diff > 0 and second.mean_diff > 0:
            adopted = candidate
    predictions = store.load_array(adopted.run_id, "test.npy")
    expected = submission(test, predictions)
    actual = pd.read_csv(store.artifact_path(adopted.run_id, "submission.csv"))
    pd.testing.assert_frame_equal(expected, actual)
    result = dict(
        candidate=candidate.run_id,
        parent=parent.run_id,
        selected=adopted.run_id,
        differences=differences,
        data_hash=summary["data_hash"],
        submission_sha256=file_sha256(store.artifact_path(adopted.run_id, "submission.csv")),
        artifacts={r.run_id: r.artifacts for r in relevant},
        rule="両seedで直接の比較元よりfold平均Accuracyが改善した場合だけ採用",
    )
    return result


def main(*, verify_existing: bool = False) -> None:
    result = build_selection()
    if verify_existing:
        saved = json.loads((OUTPUT / "selection.json").read_text(encoding="utf-8"))
        if saved != result:
            raise ValueError("既存の選定結果と再検証結果が一致しません")
    else:
        with (OUTPUT / "selection.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("candidate", "parent", "selected", "differences", "submission_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze_candidate()
    else:
        main(verify_existing=args.verify_existing)
