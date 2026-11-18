"""公式提出テンプレートから、予約・保存済みの定数提出を作る。"""

from pathlib import Path

import pandas as pd

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    store = RunStore(ROOT)
    with store.reserve("P0") as reservation:
        source = ROOT / "data" / "sample_submission.csv"
        frame = pd.read_csv(source)
        if list(frame.columns) != ["fname", "label"] or frame.fname.duplicated().any():
            raise ValueError("提出テンプレートの列またはIDが不正です")
        frame["label"] = "silence"
        record = ExperimentRecord(
            competition_id=ROOT.name,
            run_id="P0",
            metric="accuracy",
            greater_is_better=True,
            data_version="公式提出テンプレート",
            data_hash=file_sha256(source),
            split_version="p0-no-cv",
            split_hash="not-applicable",
            seed=42,
            hypothesis="定数silenceでLate Submissionの採点可否を確認する",
            changed_one_thing="提出経路の確認",
            model="constant-silence",
            params={"n_test": len(frame)},
            fold_scores=[],
            oof_score=0.0,
            code_version=code_version(REPO_ROOT),
            conclusion="CV未実施。oof_score=0は未評価を表す記録上の値で、実測Accuracyではない。",
        )
        out = store.publish(
            reservation, record, lambda path: frame.to_csv(path / "submission.csv", index=False)
        )
        print(out)


if __name__ == "__main__":
    main()
