"""run の成果物（予測・提出ファイルなど）と実験ログを、食い違わないように保存・読み込みする。

- run_id は学習の前に `reserve` で予約する（ロックファイルによる排他）。予約ごとに token を発行し、
  `publish` にはその予約（Reservation）を渡す。記録済み・成果物が残っている run_id は予約の時点で拒否する
- `publish` は一時ディレクトリに成果物を書き、ハッシュを取ってから正式な場所へ移し、最後にログへ
  1 行加える。ログの更新（`append_record` の `os.replace`）が唯一の確定点
  - ログ更新が失敗し、ログに記録が無いと確かめられたときだけ成果物を消して元に戻す
  - 記録があれば成果物は残す（確定済み）。ログを読めないなど状態が不明なら、成果物を消さずに
    残して停止する（`dev orphans` で確認できる未確定の状態）
- 強制終了などで残った「ログにない run ディレクトリ・一時ディレクトリ」は `orphans` で一覧し、
  `discard_orphans` で消す。削除は 1 件ずつ、その run_id のロックを取ってから状態を確かめ直して行う
- 読み込み時は、ログに記録したハッシュと照合する。差し替えられた成果物は読まない
"""

import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Generator, Mapping
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
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
CV_EVIDENCE = "cv_evidence.json"
QUALITY_GATE_MARKER = "quality-gates.toml"


def validate_run_id(run_id: str) -> None:
    """ファイル名・ディレクトリ名に使うので、英数字・ハイフン・アンダースコアだけを許す。"""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            f"run_id {run_id!r} は英数字で始まり、英数字・-・_ だけからなる必要があります"
        )


@contextmanager
def _exclusive(path: Path, token: str, timeout_sec: float, busy_message: str) -> Generator[None]:
    """path を排他的に作ってロックとし、抜けるときに消す。中身は token と（参考の）プロセス ID。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                owner = path.read_text(encoding="utf-8").strip() if path.exists() else "?"
                raise FileExistsError(f"{busy_message}（ロック {path}: {owner}）") from None
            time.sleep(0.05)
    try:
        os.write(fd, f"{token} pid={os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def _lock_token(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").split()[0]
    except (FileNotFoundError, IndexError):
        return None


@dataclass(frozen=True)
class Reservation:
    """`RunStore.reserve` が発行する予約。token でこの予約自身を識別する。"""

    run_id: str
    token: str


class PublishStateUnknownError(RuntimeError):
    """ログ更新が失敗し、記録が確定したか判断できない。成果物は消さずに残してある。"""


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
        validate_run_id(run_id)
        return self._locks_dir / f"{run_id}.lock"

    def records(self) -> list[ExperimentRecord]:
        return read_records(self.log_path) if self.log_path.exists() else []

    def record(self, run_id: str) -> ExperimentRecord:
        return find_record(self.records(), run_id)

    def _is_recorded(self, run_id: str) -> bool:
        return any(r.run_id == run_id for r in self.records())

    def ensure_new(self, run_id: str) -> None:
        """run_id が記録にも成果物にも使われていないことを確かめる。"""
        validate_run_id(run_id)
        if self._is_recorded(run_id):
            raise FileExistsError(
                f"run_id {run_id} は記録済みです。やり直すなら別の run_id（例: {run_id}-r2）にしてください"
            )
        if (self.runs_dir / run_id).exists():
            raise FileExistsError(
                f"{self.runs_dir / run_id} がログにない残骸として残っています。"
                "`dev orphans <番号>` で確認して削除してください"
            )

    @contextmanager
    def reserve(self, run_id: str) -> Generator[Reservation]:
        """run_id を予約する。学習から publish までをこの中で行う。"""
        reservation = Reservation(run_id=run_id, token=uuid.uuid4().hex)
        message = (
            f"run_id {run_id} は別の処理が使用中です。"
            "その処理が残っていないことを確かめてからロックファイルを削除してください"
        )
        with _exclusive(self._run_lock(run_id), reservation.token, 0.0, message):
            self.ensure_new(run_id)
            yield reservation

    def _holds(self, reservation: Reservation) -> bool:
        return _lock_token(self._run_lock(reservation.run_id)) == reservation.token

    def publish(
        self,
        reservation: Reservation,
        record: ExperimentRecord,
        write_outputs: Callable[[Path], None],
        *,
        seal: tuple[str, Callable[[Mapping[str, str]], str]] | None = None,
    ) -> Path:
        """一時ディレクトリに成果物を書き、ハッシュを記録して公開する。

        seal があれば、成果物のハッシュを渡して証跡を生成し、同じ確定処理に含める。
        """
        if (self.competition_dir / QUALITY_GATE_MARKER).is_file() and (
            seal is None or seal[0] != CV_EVIDENCE
        ):
            raise ValueError("このコンペの CV run は publish_gated_cv で公開してください")
        run_id = reservation.run_id
        if record.run_id != run_id:
            raise ValueError(f"予約 {run_id} と記録 {record.run_id} の run_id が違います")
        if not self._holds(reservation):
            raise RuntimeError(f"run_id {run_id} の予約が有効ではありません（reserve の中で呼ぶ）")
        self.ensure_new(run_id)
        tmp = self.runs_dir / f"{_TMP_PREFIX}{run_id}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        final = self.runs_dir / run_id
        try:
            write_outputs(tmp)
            artifacts = {p.name: file_sha256(p) for p in sorted(tmp.iterdir())}
            if seal is not None:
                name, content = seal
                if Path(name).name != name or name in {"", ".", ".."}:
                    raise ValueError(f"証跡ファイル名が不正です: {name}")
                path = tmp / name
                with path.open("x", encoding="utf-8") as evidence:
                    evidence.write(content(artifacts))
                artifacts[name] = file_sha256(path)
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
                reservation.token,
                _LOG_LOCK_TIMEOUT_SEC,
                "実験ログを別の処理が書き込み中です",
            ):
                append_record(self.log_path, record)
        except BaseException as error:
            self._settle_failed_publish(run_id, final, error)
            raise
        return final

    def _settle_failed_publish(self, run_id: str, final: Path, error: BaseException) -> None:
        """ログ更新の失敗後: 記録が無いと確かめられたときだけ成果物を消す。"""
        try:
            recorded = self._is_recorded(run_id)
        except Exception as read_error:
            raise PublishStateUnknownError(
                f"ログ更新に失敗し（{error!r}）、ログも読めません（{read_error!r}）。"
                f"成果物 {final} は残してあります。ログを確認してください"
            ) from error
        if not recorded:
            shutil.rmtree(final, ignore_errors=True)

    def orphans(self) -> list[Path]:
        """ログにない run ディレクトリと一時ディレクトリ（予約中の run_id のものは除く）。表示用。"""
        if not self.runs_dir.exists():
            return []
        recorded = {r.run_id for r in self.records()}
        found: list[Path] = []
        for path in sorted(self.runs_dir.iterdir()):
            if not path.is_dir() or path == self._locks_dir:
                continue
            run_id = self._run_id_of(path)
            if run_id not in recorded and not self._run_lock(run_id).exists():
                found.append(path)
        return found

    def discard_orphans(self, paths: list[Path]) -> list[Path]:
        """`orphans` で一覧したパスを消す。1 件ずつその run_id のロックを取り、状態を確かめ直す。

        一覧の後に予約・記録・パスの状態が変わっていたら、何も消さずに例外を出す。
        """
        removed: list[Path] = []
        for path in paths:
            if path.parent != self.runs_dir or path == self._locks_dir:
                raise ValueError(f"{path} は {self.runs_dir} 直下の run ディレクトリではありません")
            run_id = self._run_id_of(path)
            message = f"run_id {run_id} は予約中のため {path} を削除しません"
            with _exclusive(self._run_lock(run_id), uuid.uuid4().hex, 0.0, message):
                if self._is_recorded(run_id):
                    raise FileExistsError(
                        f"run_id {run_id} は記録済みになったため {path} を削除しません"
                    )
                if not path.is_dir():
                    raise FileNotFoundError(f"{path} はもうありません")
                shutil.rmtree(path)
            removed.append(path)
        return removed

    @staticmethod
    def _run_id_of(path: Path) -> str:
        if path.name.startswith(_TMP_PREFIX):
            return path.name.removeprefix(_TMP_PREFIX).rsplit("-", 1)[0]
        return path.name

    def artifact_path(self, run_id: str, name: str) -> Path:
        """ハッシュを照合したうえで成果物のパスを返す。"""
        validate_run_id(run_id)
        if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
            raise ValueError(f"成果物名が不正です: {name}")
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
