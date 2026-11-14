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
    """記録を追記する。同じ run_id が既にあれば拒否する（再実行するなら該当行を消す）。"""
    if log_path.exists() and any(r.run_id == record.run_id for r in read_records(log_path)):
        raise ValueError(f"run_id {record.run_id} は {log_path} に記録済みです")
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(log_path: Path) -> list[ExperimentRecord]:
    with log_path.open(encoding="utf-8") as f:
        return [ExperimentRecord(**json.loads(line)) for line in f if line.strip()]


def find_record(records: list[ExperimentRecord], run_id: str) -> ExperimentRecord:
    for record in records:
        if record.run_id == run_id:
            return record
    raise LookupError(f"run_id {run_id} の記録がありません")


@dataclass(frozen=True)
class PairedComparison:
    """同じ分割の fold どうしで比べた差 (other - base)。指標の向きは解釈側で判断する。"""

    base_run_id: str
    run_id: str
    diffs: list[float]

    @property
    def mean_diff(self) -> float:
        return sum(self.diffs) / len(self.diffs)

    @property
    def n_lower(self) -> int:
        return sum(d < 0 for d in self.diffs)

    @property
    def n_higher(self) -> int:
        return sum(d > 0 for d in self.diffs)


def compare_runs(base: ExperimentRecord, other: ExperimentRecord) -> PairedComparison:
    if base.split_version != other.split_version:
        raise ValueError(
            f"分割が異なるため比較できません: {base.split_version} と {other.split_version}"
        )
    diffs = [o - b for b, o in zip(base.fold_scores, other.fold_scores, strict=True)]
    return PairedComparison(base_run_id=base.run_id, run_id=other.run_id, diffs=diffs)
