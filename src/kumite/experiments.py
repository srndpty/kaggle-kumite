"""実験ログ (JSONL)。100コンペ共通の形式にして横断分析できるようにする。"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOG_FILE_NAME = "experiments.jsonl"


@dataclass(frozen=True, kw_only=True)
class ExperimentRecord:
    competition_id: str
    run_id: str
    data_version: str
    split_version: str
    seed: int
    hypothesis: str
    changed_one_thing: str
    model: str
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    fold_scores: list[float]
    oof_score: float
    late_score: float | None = None
    runtime_sec: float | None = None
    peak_memory_mb: float | None = None
    conclusion: str = ""
    next_experiment: str = ""


def append_record(log_path: Path, record: ExperimentRecord) -> None:
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(log_path: Path) -> list[ExperimentRecord]:
    with log_path.open(encoding="utf-8") as f:
        return [ExperimentRecord(**json.loads(line)) for line in f if line.strip()]
