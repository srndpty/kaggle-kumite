"""run の成果物（予測・提出ファイルなど）と実験ログを、食い違わないように保存・読み込みする。

- run_id は学習の前に `reserve` で予約する（ロックファイルによる排他）。同じ run_id を別プロセスが
  同時に使うことはできず、記録済み・成果物が残っている run_id は予約の時点で拒否する
- 保存（`publish`）は予約の中でだけ行える。一時ディレクトリに書き、成果物のハッシュを取ってから
  正式な場所へ移し、最後にログへ追記する。ログへの追記が公開の確定点で、追記に失敗したら
  成果物を消して元に戻す。途中で失敗しても、ログにない成果物だけが正式な場所に残ることはない
- プロセスが強制終了されたときの残骸（ログにない run ディレクトリ・一時ディレクトリ）は
  `orphans` で一覧し、`discard_orphans` で消す（`dev orphans <番号> [--delete]`）
- 読み込み時は、ログに記録したハッシュと照合する。差し替えられた成果物は読まない
"""

import os
import shutil
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
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

_TMP_PREFIX = ".tmp-"
_LOG_LOCK_TIMEOUT_SEC = 30.0


@contextmanager
def _exclusive(path: Path, timeout_sec: float, busy_message: str) -> Generator[None]:
    """path を排他的に作ってロックとし、抜けるときに消す。中身は持ち主のプロセス ID。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                owner = path.read_text(encoding="utf-8").strip() if path.exists() else "?"
                raise FileExistsError(f"{busy_message}（ロック {path}、pid {owner}）") from None
            time.sleep(0.05)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunStore:
    competition_dir: Path

    @property
    def runs_dir(self) -> Path:
        return self.competition_dir / "outputs" / "runs"

    @property
    def log_path(self) -> Path:
        return self.competition_dir / LOG_FILE_NAME

    @property
    def _locks_dir(self) -> Path:
        return self.runs_dir / ".locks"

    def _run_lock(self, run_id: str) -> Path:
        return self._locks_dir / f"{run_id}.lock"

    def records(self) -> list[ExperimentRecord]:
        return read_records(self.log_path) if self.log_path.exists() else []

    def record(self, run_id: str) -> ExperimentRecord:
        return find_record(self.records(), run_id)

    def ensure_new(self, run_id: str) -> None:
        """run_id が記録にも成果物にも使われていないことを確かめる。"""
        if any(r.run_id == run_id for r in self.records()):
            raise FileExistsError(
                f"run_id {run_id} は記録済みです。やり直すなら別の run_id（例: {run_id}-r2）にしてください"
            )
        if (self.runs_dir / run_id).exists():
            raise FileExistsError(
                f"{self.runs_dir / run_id} がログにない残骸として残っています。"
                "`dev orphans <番号>` で確認して削除してください"
            )

    @contextmanager
    def reserve(self, run_id: str) -> Generator[None]:
        """run_id を予約する。学習から publish までをこの中で行う。"""
        message = (
            f"run_id {run_id} は別のプロセスが使用中です。"
            "そのプロセスが残っていなければロックファイルを削除してください"
        )
        with _exclusive(self._run_lock(run_id), 0.0, message):
            self.ensure_new(run_id)
            yield

    def _is_reserved_here(self, run_id: str) -> bool:
        lock = self._run_lock(run_id)
        return lock.exists() and lock.read_text(encoding="utf-8").strip() == str(os.getpid())

    def publish(self, record: ExperimentRecord, write_outputs: Callable[[Path], None]) -> Path:
        """write_outputs(一時ディレクトリ) で成果物を書かせ、ハッシュを記録して公開する。"""
        run_id = record.run_id
        if not self._is_reserved_here(run_id):
            raise RuntimeError(f"run_id {run_id} を予約していません（RunStore.reserve の中で呼ぶ）")
        self.ensure_new(run_id)
        tmp = self.runs_dir / f"{_TMP_PREFIX}{run_id}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        final = self.runs_dir / run_id
        try:
            write_outputs(tmp)
            artifacts = {p.name: file_sha256(p) for p in sorted(tmp.iterdir())}
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        record = replace(
            record,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            artifacts=artifacts,
        )
        try:
            with _exclusive(
                self.log_path.with_name(self.log_path.name + ".lock"),
                _LOG_LOCK_TIMEOUT_SEC,
                "実験ログを別のプロセスが書き込み中です",
            ):
                append_record(self.log_path, record)
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            raise
        return final

    def orphans(self) -> list[Path]:
        """ログにない run ディレクトリと一時ディレクトリ（予約中の run_id のものは除く）。"""
        if not self.runs_dir.exists():
            return []
        recorded = {r.run_id for r in self.records()}
        locked = {p.stem for p in self._locks_dir.glob("*.lock")}
        found: list[Path] = []
        for path in sorted(self.runs_dir.iterdir()):
            if not path.is_dir() or path == self._locks_dir:
                continue
            if path.name.startswith(_TMP_PREFIX):
                run_id = path.name.removeprefix(_TMP_PREFIX).rsplit("-", 1)[0]
            else:
                run_id = path.name
            if run_id not in recorded and run_id not in locked:
                found.append(path)
        return found

    def discard_orphans(self) -> list[Path]:
        removed = self.orphans()
        for path in removed:
            shutil.rmtree(path)
        return removed

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
