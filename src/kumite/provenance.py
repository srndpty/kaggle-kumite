"""実験の出所（データ・分割・コード・成果物）を識別するハッシュと、実行環境の計測値。"""

import hashlib
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

_CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def files_sha256(paths: Iterable[Path]) -> str:
    """複数ファイルをまとめた1つのハッシュ。ファイル名も含めるので、中身の入れ替えも検出する。"""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.name}:{file_sha256(path)}\n".encode())
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """dtype・形状・値をまとめたハッシュ。分割（fold 番号の配列）などの同一性の確認に使う。"""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{contiguous.dtype.str}{contiguous.shape}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def code_version(repo_root: Path) -> str:
    """HEAD のコミット。未コミットの変更があれば、その差分のハッシュを `+` で付ける。"""

    def git(*args: str) -> bytes:
        # 差分には日本語などが含まれるので、ロケールの文字コードで復号せずバイト列のまま扱う。
        return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True).stdout

    head = git("rev-parse", "HEAD").decode().strip()
    diff = git("diff", "HEAD")
    if not diff:
        return head
    return f"{head}+{hashlib.sha256(diff).hexdigest()[:12]}"


def peak_memory_mb() -> float:
    """このプロセスのピークメモリ（MB）。"""
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    # Linux の ru_maxrss は KB 単位
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10
