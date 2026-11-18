"""保存済み分割を再生成し、事前の採用規則だけで最終提出を固定する。"""

import argparse
import json

import numpy as np
import pandas as pd
from dataset import OUTPUT, ROOT, load_prepared, make_folds, submission

from kumite.experiments import ExperimentRecord, compare_runs
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore


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


def main() -> None:
    store = RunStore(ROOT)
    records = store.records()
    candidate = choose_candidate(records)
    frozen = json.loads((OUTPUT / "candidate.json").read_text(encoding="utf-8"))
    if frozen["candidate"] != candidate.run_id or frozen["artifacts"] != candidate.artifacts:
        raise ValueError("確認seed前に固定した候補と一致しません")
    frame, test, _, _, summary = load_prepared()
    p5 = json.loads((ROOT / "p5_registration.json").read_text(encoding="utf-8"))
    parent_experiment = p5["parent"] if candidate.params["experiment"] == "E05" else "E01"
    parent = next(r for r in records if r.run_id == parent_experiment)
    relevant = [parent, candidate]
    if candidate.run_id != parent.run_id:
        relevant += [store.record(f"{r.params['experiment']}-s43") for r in (parent, candidate)]
    for record in relevant:
        if (
            record.data_hash != summary["data_hash"]
            or record.params["config"] != candidate.params["config"]
        ):
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
    parser.add_argument("--freeze", action="store_true")
    if parser.parse_args().freeze:
        freeze_candidate()
    else:
        main()
