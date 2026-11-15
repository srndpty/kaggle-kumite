import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_experiments import BASELINE

import kumite.experiments
import kumite.runs
from kumite.experiments import ExperimentRecord, append_record
from kumite.runs import PublishStateUnknownError, Reservation, RunStore

E01 = replace(BASELINE, run_id="E01")


def _writer(oof: np.ndarray) -> Callable[[Path], None]:
    def write(out_dir: Path) -> None:
        np.save(out_dir / "oof.npy", oof)

    return write


def _publish(store: RunStore, oof: np.ndarray, record: ExperimentRecord = BASELINE) -> Path:
    with store.reserve(record.run_id) as reservation:
        return store.publish(reservation, record, _writer(oof))


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["experiments.jsonl", "outputs"]


def test_rerun_with_same_id_changes_nothing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    before = _snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="記録済み"):
        _publish(store, np.ones((3, 2)))

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("run_id", ["../E00", "E00/x", "E00\\x", "", ".tmp", "-E00", "E 00"])
def test_unsafe_run_id_is_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"), RunStore(tmp_path).reserve(run_id):
        pass


def test_publish_requires_this_reservation(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    with pytest.raises(RuntimeError, match="予約"):
        store.publish(Reservation("E00", "forged"), BASELINE, _writer(np.zeros((3, 2))))
    with store.reserve("E00"), pytest.raises(RuntimeError, match="予約"):
        # 同じプロセスでも、この処理が取得した予約（token）でなければ拒否する
        store.publish(Reservation("E00", "other-token"), BASELINE, _writer(np.zeros((3, 2))))
    with store.reserve("E01") as reservation, pytest.raises(ValueError, match="違います"):
        store.publish(reservation, BASELINE, _writer(np.zeros((3, 2))))

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

    with pytest.raises(RuntimeError), store.reserve("E00") as reservation:
        store.publish(reservation, BASELINE, broken)

    assert store.orphans() == []
    assert store.records() == []
    store.ensure_new("E00")


def test_log_write_failure_before_commit_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新しいログを一時ファイルに書き切った後、置き換えの前に失敗する（部分書込みに相当）。"""
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    before = _snapshot(tmp_path)

    def failing_replace(src: Path, dst: Path) -> None:
        raise OSError("置き換え前の I/O エラー")

    monkeypatch.setattr(kumite.experiments.os, "replace", failing_replace)
    with pytest.raises(OSError, match="置き換え前"):
        _publish(store, np.ones((3, 2)), E01)

    assert _snapshot(tmp_path) == before  # 既存 run・ログとも不変、一時ファイルも残らない
    assert store.orphans() == []


def test_log_write_failure_after_commit_keeps_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ログに 1 行書き終えて確定した後に例外が起きる。成果物は消さない。"""
    store = RunStore(tmp_path)
    _publish(store, np.zeros((3, 2)))
    real_replace = os.replace

    def replace_then_fail(src: Path, dst: Path) -> None:
        real_replace(src, dst)
        raise OSError("確定後の I/O エラー")

    monkeypatch.setattr(kumite.experiments.os, "replace", replace_then_fail)
    with pytest.raises(OSError, match="確定後"):
        _publish(store, np.ones((3, 2)), E01)
    monkeypatch.undo()

    assert [r.run_id for r in store.records()] == ["E00", "E01"]
    np.testing.assert_array_equal(store.load_array("E01", "oof.npy"), np.ones((3, 2)))
    np.testing.assert_array_equal(store.load_array("E00", "oof.npy"), np.zeros((3, 2)))
    assert store.orphans() == []


def test_unknown_publish_state_keeps_artifacts_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ログ更新が失敗し、その後ログも読めない（確定したか判断できない）。成果物は残す。"""
    store = RunStore(tmp_path)
    real_read = kumite.runs.read_records
    append_failed = False

    def failing_append(log_path: Path, record: ExperimentRecord) -> None:
        nonlocal append_failed
        append_failed = True
        raise OSError("ディスクがいっぱい")

    def unreadable_after_failure(log_path: Path) -> list[ExperimentRecord]:
        if append_failed:
            raise OSError("ログを読めない")
        return real_read(log_path)

    monkeypatch.setattr(kumite.runs, "append_record", failing_append)
    monkeypatch.setattr(kumite.runs, "read_records", unreadable_after_failure)
    store.log_path.write_text("", encoding="utf-8")  # ログはあるが、失敗後に読めなくなる
    with pytest.raises(PublishStateUnknownError, match="残してあります"):
        _publish(store, np.zeros((3, 2)))
    monkeypatch.undo()

    assert (store.runs_dir / "E00").is_dir()
    assert store.orphans() == [store.runs_dir / "E00"]  # 回復対象として見える


def test_truncated_log_is_not_appended(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"
    log_path.write_bytes(b'{"competition_id": "001", "run_')

    with pytest.raises(ValueError, match="途中で切れた"):
        append_record(log_path, BASELINE)

    assert log_path.read_bytes() == b'{"competition_id": "001", "run_'


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
    assert store.discard_orphans(store.orphans()) == [tmp_leftover, leftover]
    assert store.orphans() == []
    assert (store.runs_dir / "E00").exists()


def test_discard_refuses_when_reserved_after_listing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    leftover = store.runs_dir / ".tmp-E01-0123abcd"
    leftover.mkdir(parents=True)
    listed = store.orphans()

    with store.reserve("E01"), pytest.raises(FileExistsError, match="予約中"):
        store.discard_orphans(listed)

    assert leftover.is_dir()


def test_discard_refuses_when_recorded_after_listing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    leftover = store.runs_dir / "E01"
    leftover.mkdir(parents=True)
    listed = store.orphans()
    append_record(store.log_path, E01)

    with pytest.raises(FileExistsError, match="記録済み"):
        store.discard_orphans(listed)

    assert leftover.is_dir()


def test_discard_rejects_paths_outside_runs_dir(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    outside = tmp_path / "data"
    outside.mkdir()

    with pytest.raises(ValueError, match="直下"):
        store.discard_orphans([outside])

    assert outside.is_dir()


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
