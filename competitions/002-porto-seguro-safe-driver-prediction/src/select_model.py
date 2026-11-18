"""外部評価前に、事前規則で最終 run を固定する。"""

import json

import numpy as np
from dataset import COMPETITION_DIR, DATA_FILES
from pipeline import STORE

from kumite.experiments import compare_runs
from kumite.provenance import file_sha256, files_sha256


def select() -> dict[str, object]:
    selected = "E01"
    comparisons = []
    for parent, candidate in (("E01", "E04"), ("E04", "E06")):
        if selected != parent:
            break
        diffs = []
        for seed in (42, 43, 44):
            suffix = "" if seed == 42 else f"-s{seed}"
            base, other = STORE.record(parent + suffix), STORE.record(candidate + suffix)
            if (
                base.seed != seed
                or other.seed != seed
                or base.data_hash != files_sha256(DATA_FILES)
            ):
                raise ValueError("現在のデータまたは seed が比較対象と違います")
            for record in (base, other):
                STORE.load_array(record.run_id, "oof.npy")
                STORE.artifact_path(record.run_id, "submission.csv")
            diffs.append(compare_runs(base, other).mean_diff)
        accepted = float(np.mean(diffs)) > 0 and sum(d > 0 for d in diffs) >= 2
        comparisons.append(
            {
                "parent": parent,
                "candidate": candidate,
                "seed_diffs": diffs,
                "mean_diff": float(np.mean(diffs)),
                "accepted": accepted,
            }
        )
        if accepted:
            selected = candidate
    submission_path = STORE.artifact_path(selected, "submission.csv")
    result: dict[str, object] = {
        "selected_run": selected,
        "comparisons": comparisons,
        "submission_sha256": file_sha256(submission_path),
        "rule": "3 seed のfold平均Gini差が正、2 seed以上で改善。提出はseed42の5fold平均",
        "experiment_log_sha256": file_sha256(STORE.log_path),
    }
    output = COMPETITION_DIR / "outputs" / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(select(), ensure_ascii=False, indent=2))
