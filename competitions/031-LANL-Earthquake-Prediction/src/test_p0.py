from pathlib import Path

import pytest
from p0 import write_dummy


def test_dummy_follows_sample_order(tmp_path: Path) -> None:
    (tmp_path / "train.csv").touch()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for seg_id in ("seg_a", "seg_b"):
        (test_dir / f"{seg_id}.csv").touch()
    (tmp_path / "sample_submission.csv").write_text(
        "seg_id,time_to_failure\nseg_b,1\nseg_a,2\n", encoding="utf-8"
    )

    output = tmp_path / "output" / "dummy.csv"
    write_dummy(tmp_path, output)

    assert output.read_text(encoding="utf-8") == "seg_id,time_to_failure\nseg_b,0.0\nseg_a,0.0\n"


@pytest.mark.parametrize("ids", ["seg_a\nseg_a\n", "seg_b\n", "\n"])
def test_dummy_rejects_invalid_sample_ids(tmp_path: Path, ids: str) -> None:
    (tmp_path / "train.csv").touch()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "seg_a.csv").touch()
    (tmp_path / "sample_submission.csv").write_text(
        "seg_id,time_to_failure\n" + "".join(f"{seg_id},0\n" for seg_id in ids.splitlines()),
        encoding="utf-8",
    )

    output = tmp_path / "dummy.csv"
    with pytest.raises(ValueError, match="ID"):
        write_dummy(tmp_path, output)
    assert not output.exists()
