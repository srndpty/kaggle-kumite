"""外部評価前に、seed42の最良候補をseed43で確認して固定する。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from cv import make_folds, split_hash
from dataset import COMPETITION_DIR, DATA_FILES, KEY, load_data
from pipeline import EXPERIMENTS, STORE, parameters

from kumite.experiments import ExperimentRecord, compare_runs
from kumite.provenance import file_sha256, files_sha256
from kumite.runs import RunStore


def validate_record(record: ExperimentRecord, experiment: str, seed: int) -> None:
    expected = EXPERIMENTS[experiment]
    if (
        record.competition_id != "004"
        or record.metric != "roc_auc"
        or not record.greater_is_better
        or record.seed != seed
        or record.params.get("experiment") != experiment
        or record.params.get("definition") != asdict(expected)
        or record.params.get("benchmark") is not False
        or record.model != ("prior" if expected.prior else "lightgbm_binary")
        or record.data_version
        != f"application-v1|history={expected.history}|missing_count={expected.missing_count}"
        or any(record.params.get(k) != v for k, v in parameters(seed).items())
        or len(record.fold_scores) != 3
    ):
        raise ValueError("実験定義または評価条件が一致しません")


def select(
    candidate: str, parent: str, store: RunStore = STORE, output: Path | None = None
) -> dict[str, object]:
    if EXPERIMENTS[candidate].parent != parent or candidate == "E00":
        raise ValueError("候補と直接の比較元を指定してください")
    data_hash = files_sha256(DATA_FILES)
    train, _, _ = load_data()
    y = train["TARGET"].to_numpy(dtype=np.int64)
    expected_splits = {seed: make_folds(train, y, seed) for seed in (42, 43)}

    def validate_split(record: ExperimentRecord) -> None:
        folds, version = expected_splits[record.seed]
        if record.split_version != version or record.split_hash != split_hash(folds):
            raise ValueError("現在のデータから再生成した分割とrun記録が一致しません")
        saved = pd.read_csv(store.artifact_path(record.run_id, "folds.csv"))
        if (
            list(saved.columns) != [KEY, "fold"]
            or not np.array_equal(saved[KEY], train[KEY])
            or not np.array_equal(saved["fold"], folds)
        ):
            raise ValueError("保存された分割のID・行順・foldが現在の分割と一致しません")

    # 同じseed42に存在する本実験をすべて比較し、最良候補以外を持ち込ませない。
    eligible = []
    for record in store.records():
        experiment = record.params.get("experiment")
        if (
            record.seed == 42
            and not record.params.get("benchmark")
            and experiment in EXPERIMENTS
            and experiment != "E00"
        ):
            validate_record(record, str(experiment), 42)
            if record.data_hash != data_hash:
                raise ValueError("現在のデータと異なります")
            validate_split(record)
            eligible.append(record)
    if not eligible:
        raise ValueError("seed42の本実験がありません")
    for record in eligible[1:]:
        compare_runs(eligible[0], record)
    best = min(eligible, key=lambda r: (-float(np.mean(r.fold_scores)), float(r.runtime_sec or 0)))
    if best.params["experiment"] != candidate:
        raise ValueError(f"seed42の最良候補は {best.run_id} です")
    differences = []
    pair_runs = []
    for seed in (42, 43):
        pair = []
        for experiment in (parent, candidate):
            matches = [
                r
                for r in store.records()
                if r.seed == seed
                and r.params.get("experiment") == experiment
                and r.params.get("benchmark") is False
            ]
            if len(matches) != 1:
                raise ValueError("各実験・seedに一意な本実験runが必要です")
            record = matches[0]
            validate_record(record, experiment, seed)
            if record.data_hash != data_hash:
                raise ValueError("現在のデータと異なります")
            validate_split(record)
            for name in ("oof.npy", "test.npy", "folds.csv", "submission.csv", "features.json"):
                store.artifact_path(record.run_id, name)
            pair.append(record)
        differences.append(compare_runs(pair[0], pair[1]).mean_diff)
        pair_runs.append(pair)
    accepted = all(d > 0 for d in differences)
    selected = pair_runs[0][1 if accepted else 0]
    result: dict[str, object] = {
        "candidate": candidate,
        "parent": parent,
        "seed_differences": differences,
        "accepted": accepted,
        "selected_run": selected.run_id,
        "rule": "seed42最良候補を直接の比較元とseed43で確認。両seedで改善時に採用。seed42の3fold平均を提出",
        "submission_sha256": file_sha256(store.artifact_path(selected.run_id, "submission.csv")),
        "experiment_log_sha256": file_sha256(store.log_path),
    }
    destination = output or COMPETITION_DIR / "outputs" / "selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=EXPERIMENTS)
    parser.add_argument("parent", choices=EXPERIMENTS)
    args = parser.parse_args()
    print(json.dumps(select(args.candidate, args.parent), ensure_ascii=False, indent=2))
