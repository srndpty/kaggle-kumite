"""run の成果物（予測・提出ファイルなど）と実験ログを、食い違わないように保存・読み込みする。

- 保存は fail-closed: 一時ディレクトリに書き、成果物のハッシュを取ってから正式な場所へ移し、
  最後にログへ追記する。途中で失敗すれば一時ディレクトリごと消し、既存の run には触れない
- 既存の run_id（ログ・ディレクトリのどちらか）への保存は、学習を始める前に拒否する
- 読み込み時は、ログに記録したハッシュと照合する。差し替えられた成果物は読まない
"""

import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from kumite.experiments import (
    LOG_FILE_NAME,
    ExperimentRecord,
    append_record,
    find_record,
    read_records,
)
from kumite.provenance import file_sha256


@dataclass(frozen=True)
class RunStore:
    competition_dir: Path

    @property
    def runs_dir(self) -> Path:
        return self.competition_dir / "outputs" / "runs"

    @property
    def log_path(self) -> Path:
        return self.competition_dir / LOG_FILE_NAME

    def records(self) -> list[ExperimentRecord]:
        return read_records(self.log_path) if self.log_path.exists() else []

    def record(self, run_id: str) -> ExperimentRecord:
        return find_record(self.records(), run_id)

    def ensure_new(self, run_id: str) -> None:
        """run_id が未使用であることを確かめる。学習の前に呼ぶ。"""
        if any(r.run_id == run_id for r in self.records()):
            raise FileExistsError(
                f"run_id {run_id} は記録済みです。やり直すなら別の run_id（例: {run_id}-r2）にしてください"
            )
        if (self.runs_dir / run_id).exists():
            raise FileExistsError(
                f"{self.runs_dir / run_id} が既にあります（ログにない中断した run の残骸なら確認して削除）"
            )

    def publish(self, record: ExperimentRecord, write_outputs: Callable[[Path], None]) -> Path:
        """write_outputs(一時ディレクトリ) で成果物を書かせ、ハッシュを記録して公開する。"""
        self.ensure_new(record.run_id)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.runs_dir / f".tmp-{record.run_id}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        final = self.runs_dir / record.run_id
        try:
            write_outputs(tmp)
            artifacts = {p.name: file_sha256(p) for p in sorted(tmp.iterdir())}
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        append_record(
            self.log_path,
            replace(
                record,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                artifacts=artifacts,
            ),
        )
        return final

    def artifact_path(self, run_id: str, name: str) -> Path:
        """ハッシュを照合したうえで成果物のパスを返す。"""
        expected = self.record(run_id).artifacts.get(name)
        if expected is None:
            raise LookupError(f"run {run_id} の成果物に {name} はありません")
        path = self.runs_dir / run_id / name
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path} のハッシュがログと一致しません（差し替えられた可能性）")
        return path

    def load_array(self, run_id: str, name: str) -> np.ndarray:
        return np.load(self.artifact_path(run_id, name))

    def input_hashes(self, run_id: str, *names: str) -> dict[str, str]:
        """他の run の成果物を入力に使うときに、記録用の {"run_id/name": ハッシュ} を返す。"""
        artifacts = self.record(run_id).artifacts
        return {f"{run_id}/{name}": artifacts[name] for name in names}
