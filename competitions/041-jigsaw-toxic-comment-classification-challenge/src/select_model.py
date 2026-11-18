"""seed 42・43のペア比較で採否を固定し、提出前の選定証跡を保存する。"""

import argparse
import json

from analyze import compare
from dataset import COMPETITION_DIR, DATA_FILES
from pipeline import EXPERIMENTS, STORE

from kumite.experiments import compare_runs
from kumite.provenance import file_sha256, files_sha256


def select(candidate: str, parent: str):
    if EXPERIMENTS[candidate][1] != parent:
        raise ValueError("候補の直接の比較元を指定してください")
    comparisons = [compare([parent, candidate]), compare([f"{parent}-s43", f"{candidate}-s43"])]
    pairs = [
        (STORE.record(parent), STORE.record(candidate)),
        (STORE.record(f"{parent}-s43"), STORE.record(f"{candidate}-s43")),
    ]
    for seed, pair in zip((42, 43), pairs, strict=True):
        if any(
            r.seed != seed
            or r.competition_id != "041"
            or r.metric != "macro_auc"
            or not r.greater_is_better
            for r in pair
        ):
            raise ValueError("選定に必要なseedが不一致です")
    for pair in pairs:
        for record in pair:
            if record.data_hash != files_sha256(DATA_FILES):
                raise ValueError("現在のデータとrunが異なります")
            for artifact in record.artifacts:
                STORE.artifact_path(record.run_id, artifact)
    if any(r.runtime_sec is None for pair in pairs for r in pair):
        raise ValueError("実行時間がありません")

    for experiment, record in ((parent, pairs[0][0]), (candidate, pairs[0][1])):
        if (
            record.params.get("experiment") != experiment
            or record.params.get("parent") != EXPERIMENTS[experiment][1]
        ):
            raise ValueError("保存済み実験の定義が一致しません")

    def signature(record):
        return (
            record.model,
            record.data_version,
            record.params,
            {k: v for k, v in record.inputs.items() if k.startswith("pretrained/")},
        )

    if signature(pairs[0][0]) != signature(pairs[1][0]) or signature(pairs[0][1]) != signature(
        pairs[1][1]
    ):
        raise ValueError("seed以外のモデル設定が一致しません")
    deltas = [compare_runs(a, b).mean_diff for a, b in pairs]
    accept = all(d > 0 for d in deltas)
    if all(d == 0 for d in deltas):
        accept = sum(r.runtime_sec or 0 for _, r in pairs) < sum(
            r.runtime_sec or 0 for r, _ in pairs
        )
    selected = candidate if accept else parent
    path = STORE.artifact_path(selected, "submission.csv")
    report = {
        "selected": selected,
        "candidate": candidate,
        "parent": parent,
        "comparisons": comparisons,
        "submission_sha256": file_sha256(path),
        "experiment_log_sha256": file_sha256(STORE.log_path),
        "rule": "両seedでfold平均macro AUCが改善。同点は実行時間で選択。Lateは使用しない",
    }
    target = COMPETITION_DIR / "outputs" / "selection.json"
    if target.exists():
        raise FileExistsError("選定済みです。既存の選定証跡は上書きしません")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("parent")
    a = p.parse_args()
    print(json.dumps(select(a.candidate, a.parent), ensure_ascii=False, indent=2))
