import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_experiments import BASELINE

import kumite.runs
from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore


def _writer(oof: np.ndarray) -> Callable[[Path], None]:
    def write(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", oof)

    return write


def _publish(store: RunStore, oof: np.ndarray, record: ExperimentRecord = BASELINE) -> Path:
    with store.reserve(record.run_id):
        return store.publish(record, _writer(oof))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_publish_records_artifact_hashes(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    oof = np.arange(6, dtype=np.float64).reshape(3, 2)

    final = _publish(store, oof)

    record = store.record("E00")
    assert set(record.artifacts) == {"oof.npy"}
    assert record.created_at
    assert final == store.runs_dir / "E00"
    np.testing.assert_array_equal(store.load_array("E00", "oof.npy"), oof)
    assert store.orphans() == []
    assert not list((store.runs_dir / ".locks").iterdir())


def test_rerun_with_same_id_changes_nothing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    before = _snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="記録済み"):
        _publish(store, np.ones((3, 2)))

    assert _snapshot(tmp_path) == before


def test_publish_requires_reservation(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    with pytest.raises(RuntimeError, match="予約"):
        store.publish(BASELINE, _writer(np.zeros((3, 2))))

    assert store.records() == []


def test_second_process_cannot_reserve_same_run_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    other_process = (
        "import sys; from pathlib import Path; from kumite.runs import RunStore\n"
        "with RunStore(Path(sys.argv[1])).reserve('E00'):\n"
        "    pass\n"
    )

    with store.reserve("E00"):
        result = subprocess.run(
            [sys.executable, "-c", other_process, str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    assert result.returncode != 0
    assert "使用中" in result.stderr
    with store.reserve("E00"):  # 予約は解放されている
        pass


def test_failed_write_leaves_no_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    def broken(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", np.zeros(3))
        raise RuntimeError("学習途中の失敗")

    with pytest.raises(RuntimeError), store.reserve("E00"):
        store.publish(BASELINE, broken)

    assert list(store.runs_dir.iterdir()) == [store.runs_dir / ".locks"]
    assert store.records() == []
    store.ensure_new("E00")


def test_log_append_failure_rolls_back_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)

    def failing_append(log_path: Path, record: ExperimentRecord) -> None:
        raise OSError("ディスクがいっぱい")

    monkeypatch.setattr(kumite.runs, "append_record", failing_append)

    with pytest.raises(OSError, match="ディスク"):
        _publish(store, np.zeros((3, 2)))

    assert not (store.runs_dir / "E00").exists()
    assert store.orphans() == []
    store.ensure_new("E00")


def test_orphans_are_listed_and_discarded(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    leftover = store.runs_dir / "E01"
    leftover.mkdir()
    tmp_leftover = store.runs_dir / ".tmp-E01-r2-0123abcd"
    tmp_leftover.mkdir()

    with pytest.raises(FileExistsError, match="残骸"):
        store.ensure_new("E01")
    with store.reserve("E02"):
        reserved_tmp = store.runs_dir / ".tmp-E02-89abcdef"
        reserved_tmp.mkdir()
        assert store.orphans() == [tmp_leftover, leftover]  # 予約中の E02 は残骸ではない

    reserved_tmp.rmdir()
    assert store.discard_orphans() == [tmp_leftover, leftover]
    assert store.orphans() == []
    assert (store.runs_dir / "E00").exists()


def test_replaced_artifact_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    np.save(store.runs_dir / "E00" / "oof.npy", np.ones((3, 2)))

    with pytest.raises(ValueError, match="ハッシュ"):
        store.load_array("E00", "oof.npy")


def test_input_hashes_name_the_source_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))

    hashes = store.input_hashes("E00", "oof.npy")

    assert hashes == {"E00/oof.npy": store.record("E00").artifacts["oof.npy"]}
