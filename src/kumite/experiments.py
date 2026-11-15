"""実験ログ (JSONL)。100コンペ共通の形式にして横断分析できるようにする。

記録は追記のみで、計測した値は書き換えない。後から書き足してよいのは late_score・conclusion・
next_experiment だけ。同じ実験をやり直すときは `E01-r2` のように別の run_id にする。
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOG_FILE_NAME = "experiments.jsonl"


@dataclass(frozen=True, kw_only=True)
class ExperimentRecord:
    competition_id: str
    run_id: str
    created_at: str = ""  # RunStore.publish が記録する
    # 何で評価したか（compare_runs はこれらが一致する run だけを比べる）
    metric: str
    greater_is_better: bool
    data_version: str  # 特徴量・前処理の名前
    data_hash: str  # 入力データファイルのハッシュ
    split_version: str
    split_hash: str  # fold 番号の配列のハッシュ
    seed: int
    # 何を試したか
    hypothesis: str
    changed_one_thing: str
    model: str
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    # 結果（丸めずに保存し、表示するときだけ丸める）
    fold_scores: list[float]
    oof_score: float
    late_score: float | None = None
    runtime_sec: float | None = None
    peak_memory_mb: float | None = None
    # 再現のための証跡
    code_version: str
    inputs: dict[str, str] = field(
        default_factory=dict[str, str]
    )  # 使った他 run の成果物 → ハッシュ
    artifacts: dict[str, str] = field(default_factory=dict[str, str])  # 成果物 → ハッシュ
    # 解釈
    conclusion: str = ""
    next_experiment: str = ""


def append_record(log_path: Path, record: ExperimentRecord) -> None:
    """記録を 1 行加える。同じ run_id が既にあれば拒否する。

    追記ではなく「既存の内容 + 新しい行」を一時ファイルに書き、`os.replace` で置き換える。
    置き換えが唯一の確定点で、それより前に失敗すればログは元のまま。書きかけの行は残らない。
    """
    existing = log_path.read_bytes() if log_path.exists() else b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError(f"{log_path} の末尾が改行で終わっていません（途中で切れた可能性）")
    if existing and any(r.run_id == record.run_id for r in read_records(log_path)):
        raise ValueError(f"run_id {record.run_id} は {log_path} に記録済みです")
    line = (json.dumps(asdict(record), ensure_ascii=False) + "\n").encode("utf-8")
    tmp = log_path.with_name(f".{log_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(existing + line)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(log_path)
    finally:
        tmp.unlink(missing_ok=True)


def read_records(log_path: Path) -> list[ExperimentRecord]:
    with log_path.open(encoding="utf-8") as f:
        return [ExperimentRecord(**json.loads(line)) for line in f if line.strip()]


def find_record(records: list[ExperimentRecord], run_id: str) -> ExperimentRecord:
    for record in records:
        if record.run_id == run_id:
            return record
    raise LookupError(f"run_id {run_id} の記録がありません")


# 同じ評価条件とみなすために一致が必要な項目
_COMPARABLE_FIELDS = (
    "competition_id",
    "metric",
    "greater_is_better",
    "data_hash",
    "split_version",
    "split_hash",
)


@dataclass(frozen=True)
class PairedComparison:
    """同じ分割の fold どうしで比べた差 (other - base)。"""

    base_run_id: str
    run_id: str
    diffs: list[float]
    greater_is_better: bool

    @property
    def mean_diff(self) -> float:
        return sum(self.diffs) / len(self.diffs)

    @property
    def n_improved(self) -> int:
        return sum((d > 0) if self.greater_is_better else (d < 0) for d in self.diffs)

    @property
    def n_worsened(self) -> int:
        return sum((d < 0) if self.greater_is_better else (d > 0) for d in self.diffs)


def compare_runs(base: ExperimentRecord, other: ExperimentRecord) -> PairedComparison:
    """評価条件（データ・分割・指標）が一致する run だけを fold ごとに比べる。"""
    mismatched = [
        f"{name}: {getattr(base, name)} と {getattr(other, name)}"
        for name in _COMPARABLE_FIELDS
        if getattr(base, name) != getattr(other, name)
    ]
    if mismatched:
        raise ValueError("評価条件が異なるため比較できません: " + " / ".join(mismatched))
    diffs = [o - b for b, o in zip(base.fold_scores, other.fold_scores, strict=True)]
    return PairedComparison(
        base_run_id=base.run_id,
        run_id=other.run_id,
        diffs=diffs,
        greater_is_better=base.greater_is_better,
    )
