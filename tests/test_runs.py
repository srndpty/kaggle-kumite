from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_experiments import BASELINE

from kumite.runs import RunStore


def _writer(oof: np.ndarray) -> Callable[[Path], None]:
    def write(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", oof)

    return write


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_publish_records_artifact_hashes(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    oof = np.arange(6, dtype=np.float64).reshape(3, 2)

    final = store.publish(BASELINE, _writer(oof))

    record = store.record("E00")
    assert set(record.artifacts) == {"oof.npy"}
    assert record.created_at
    assert final == store.runs_dir / "E00"
    np.testing.assert_array_equal(store.load_array("E00", "oof.npy"), oof)
    assert [p.name for p in store.runs_dir.iterdir()] == ["E00"]


def test_rerun_with_same_id_changes_nothing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.publish(BASELINE, _writer(np.zeros((3, 2))))
    before = _snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="E00"):
        store.publish(BASELINE, _writer(np.ones((3, 2))))

    assert _snapshot(tmp_path) == before


def test_failed_write_leaves_no_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    def broken(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", np.zeros(3))
        raise RuntimeError("学習途中の失敗")

    with pytest.raises(RuntimeError):
        store.publish(BASELINE, broken)

    assert list(store.runs_dir.iterdir()) == []
    assert store.records() == []
    store.ensure_new("E00")


def test_leftover_directory_blocks_reuse_of_run_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (store.runs_dir / "E00").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="残骸"):
        store.ensure_new("E00")


def test_replaced_artifact_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.publish(BASELINE, _writer(np.zeros((3, 2))))
    np.save(store.runs_dir / "E00" / "oof.npy", np.ones((3, 2)))

    with pytest.raises(ValueError, match="ハッシュ"):
        store.load_array("E00", "oof.npy")


def test_input_hashes_name_the_source_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.publish(BASELINE, _writer(np.zeros((3, 2))))

    hashes = store.input_hashes("E00", "oof.npy")

    assert hashes == {"E00/oof.npy": store.record("E00").artifacts["oof.npy"]}
