"""P0 の定数提出を生成する。"""

import argparse
from pathlib import Path

import pandas as pd
from dataset import ID, TARGET, check_ids

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="E00")
    args = parser.parse_args()
    generate(ROOT, args.run_id)


def generate(root: Path, run_id: str) -> None:
    store = RunStore(root)
    source = root / "data" / "sample_submission.csv"
    with store.reserve(run_id) as reservation:
        sample = pd.read_csv(source)
        check_ids(sample)
        if list(sample) != [ID, TARGET]:
            raise ValueError("提出テンプレートの列が一致しません")
        sample[TARGET] = 0.5
        record = ExperimentRecord(
            competition_id="7",
            run_id=reservation.run_id,
            metric="ROC AUC",
            greater_is_better=True,
            data_version="sample-v1",
            data_hash=file_sha256(source),
            split_version="p0",
            split_hash="",
            seed=42,
            hypothesis="定数予測で公式採点を確認",
            changed_one_thing="定数 0.5",
            model="constant",
            fold_scores=[],
            oof_score=0.0,  # P0 は CV を実施しない。
            code_version=code_version(REPO_ROOT),
        )

        def write(out: Path) -> None:
            sample.to_csv(out / "submission.csv", index=False)

        print(store.publish(reservation, record, write))


if __name__ == "__main__":
    main()
