from pathlib import Path

import pytest

from kumite.catalog import find_competition, load_catalog
from kumite.paths import CATALOG_CSV, TEMPLATE_DIR
from kumite.scaffold import create_competition_dir


def test_create_competition_dir_renders_templates(tmp_path: Path) -> None:
    competition = find_competition(load_catalog(CATALOG_CSV), 1)

    target = create_competition_dir(competition, tmp_path, TEMPLATE_DIR)

    assert target == tmp_path / competition.dir_name
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert competition.title in readme
    assert competition.url in readme
    assert "$" not in readme
    for name in ("src", "notebooks", "data", "outputs"):
        assert (target / name).is_dir()


def test_create_competition_dir_refuses_to_overwrite(tmp_path: Path) -> None:
    competition = find_competition(load_catalog(CATALOG_CSV), 1)
    create_competition_dir(competition, tmp_path, TEMPLATE_DIR)

    with pytest.raises(FileExistsError):
        create_competition_dir(competition, tmp_path, TEMPLATE_DIR)
