"""CV の評価条件とソースの証跡を照合する。"""

from hashlib import sha256
from pathlib import Path

from kumite.experiments import ExperimentRecord

SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_FILES = ("steel.py", "model.py", "train.py", "split.py", "cv_checks.py")


def read_sources(directory: Path = SOURCE_DIR) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in SOURCE_FILES}


def source_hashes(sources: dict[str, bytes]) -> dict[str, str]:
    return {f"source/{name}": sha256(content).hexdigest() for name, content in sources.items()}


def verify_sources(record: ExperimentRecord, hashes: dict[str, str]) -> None:
    if any(record.inputs.get(name) != digest for name, digest in hashes.items()):
        raise ValueError(
            f"{record.run_id} の CV ソースが未記録または変更されています。新しい run ID で再評価してください"
        )


def evaluation_key(record: ExperimentRecord) -> tuple[object, ...]:
    return (
        record.competition_id,
        record.metric,
        record.greater_is_better,
        record.data_hash,
        record.split_version,
        record.split_hash,
        record.seed,
    )
