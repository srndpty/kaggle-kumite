import subprocess
from pathlib import Path

import numpy as np

from kumite.provenance import array_sha256, code_version, files_sha256


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_code_version_marks_uncommitted_non_ascii_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-m", "init")
    clean = code_version(tmp_path)

    target.write_text("x = 2  # 日本語のコメント\n", encoding="utf-8")
    dirty = code_version(tmp_path)

    assert "+" not in clean
    assert dirty.startswith(clean + "+")


def test_array_hash_distinguishes_order_and_dtype() -> None:
    a = np.array([0, 1, 2], dtype=np.int8)

    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a[::-1])
    assert array_sha256(a) != array_sha256(a.astype(np.int16))


def test_files_hash_detects_swapped_contents(tmp_path: Path) -> None:
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    train.write_text("1", encoding="utf-8")
    test.write_text("2", encoding="utf-8")
    before = files_sha256([train, test])

    train.write_text("2", encoding="utf-8")
    test.write_text("1", encoding="utf-8")

    assert files_sha256([train, test]) != before
