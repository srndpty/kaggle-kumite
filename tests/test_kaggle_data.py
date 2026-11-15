"""取得したデータの展開。"""

import zipfile
from pathlib import Path

from kumite.kaggle_data import extract_archives


def test_extract_archives_unpacks_nested_zips(tmp_path: Path) -> None:
    inner = tmp_path / "a.csv.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("a.csv", "x\n1\n")
    outer = tmp_path / "competition.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, inner.name)
        zf.writestr("b.csv", "y\n2\n")
    inner.unlink()

    extract_archives(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "b.csv"]
    assert (tmp_path / "a.csv").read_text() == "x\n1\n"
