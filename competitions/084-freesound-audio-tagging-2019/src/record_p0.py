"""Kaggle で採点が完了した P0 の証跡を保存する。"""

import argparse
import shutil
from pathlib import Path

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]


def record_p0(kernel: Path, evidence: Path, public: float, private: float) -> Path:
    if not (0 <= public <= 1 and 0 <= private <= 1):
        raise ValueError("P0 scores must be in [0, 1]")
    sample = ROOT / "data" / "sample_submission.csv"
    store = RunStore(ROOT)
    with store.reserve("P0") as reservation:
        record = ExperimentRecord(
            competition_id=ROOT.name,
            run_id="P0",
            metric="lwlrap",
            greater_is_better=True,
            data_version="公式提出テンプレート",
            data_hash=file_sha256(sample),
            split_version="p0-no-cv",
            split_hash="not-applicable",
            seed=42,
            hypothesis="固定タグ順位で Late Submission の採点可否を確認する",
            changed_one_thing="提出経路の確認",
            model="constant-tag-ranking",
            params={"public_lwlrap": public, "private_lwlrap": private},
            fold_scores=[],
            oof_score=0.0,
            late_score=private,
            code_version=code_version(REPO_ROOT),
            conclusion="CV 未実施。oof_score のゼロは記録上の未採点値。",
        )

        def write(path: Path) -> None:
            for name in ("submit.ipynb", "kernel-metadata.json"):
                shutil.copyfile(kernel / name, path / name)
            shutil.copyfile(evidence, path / "kaggle-submission.txt")

        return store.publish(reservation, record, write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("public", type=float)
    parser.add_argument("private", type=float)
    args = parser.parse_args()
    print(record_p0(args.kernel, args.evidence, args.public, args.private))
