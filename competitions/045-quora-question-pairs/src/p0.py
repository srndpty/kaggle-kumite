"""定数予測で公式の採点経路を確認する。"""

from pathlib import Path

import pandas as pd

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "data" / "sample_submission.csv"
    store = RunStore(ROOT)
    with store.reserve("E00") as reservation:
        frame = pd.read_csv(source)
        if list(frame.columns) != ["test_id", "is_duplicate"]:
            raise ValueError("公式提出見本の列が不正です")
        test_ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["test_id"], nrows=len(frame))[
            "test_id"
        ]
        ids = frame["test_id"]
        if ids.isna().any() or ids.duplicated().any() or not ids.equals(test_ids):
            raise ValueError("公式提出見本とテストの ID が一致しません")
        frame["is_duplicate"] = 0.5
        record = ExperimentRecord(
            competition_id=ROOT.name,
            run_id="E00",
            metric="log_loss",
            greater_is_better=False,
            data_version="公式提出テンプレート",
            data_hash=file_sha256(source),
            split_version="p0-no-cv",
            split_hash="not-applicable",
            seed=42,
            hypothesis="定数予測で公式採点が完了する",
            changed_one_thing="提出経路の確認",
            model="constant-0.5",
            params={"n_test": len(frame)},
            fold_scores=[],
            oof_score=0.0,
            code_version=code_version(REPO_ROOT),
            conclusion="CV 未実施。oof_score の 0 は未評価を表す記録上の値。",
        )
        store.publish(
            reservation, record, lambda path: frame.to_csv(path / "submission.csv", index=False)
        )


if __name__ == "__main__":
    main()
