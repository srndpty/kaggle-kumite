import json
from pathlib import Path

import pytest

from kumite.kaggle_notebook import validate_cpu_notebook, write_cpu_notebook

KERNEL = {"kernel_id": "test/kernel", "title": "Kernel", "competition": "sample"}
SOURCE = "print('valid')\n"


def test_cpu_notebook_accepts_generated_payload(tmp_path: Path) -> None:
    write_cpu_notebook(tmp_path, SOURCE, **KERNEL)
    assert validate_cpu_notebook(tmp_path, SOURCE, **KERNEL) == SOURCE


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("is_private", False),
        ("enable_internet", True),
        ("enable_gpu", True),
        ("competition_sources", ["other"]),
        ("id", "other/kernel"),
    ],
)
def test_cpu_notebook_rejects_metadata(tmp_path: Path, key: str, value: object) -> None:
    write_cpu_notebook(tmp_path, SOURCE, **KERNEL)
    path = tmp_path / "kernel-metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_cpu_notebook(tmp_path, SOURCE, **KERNEL)


@pytest.mark.parametrize("extra_cell", [False, True])
def test_cpu_notebook_rejects_modified_code(tmp_path: Path, extra_cell: bool) -> None:
    write_cpu_notebook(tmp_path, SOURCE, **KERNEL)
    path = tmp_path / "submit.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if extra_cell:
        notebook["cells"].append(notebook["cells"][0].copy())
    else:
        notebook["cells"][0]["source"] = ["print('changed')\n"]
    path.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_cpu_notebook(tmp_path, SOURCE, **KERNEL)


def test_cpu_notebook_rejects_invalid_syntax(tmp_path: Path) -> None:
    write_cpu_notebook(tmp_path, "if:", **KERNEL)
    with pytest.raises(SyntaxError):
        validate_cpu_notebook(tmp_path, "if:", **KERNEL)


@pytest.mark.parametrize("name", ["submit.ipynb", "kernel-metadata.json"])
def test_cpu_notebook_requires_files(tmp_path: Path, name: str) -> None:
    write_cpu_notebook(tmp_path, SOURCE, **KERNEL)
    (tmp_path / name).unlink()
    with pytest.raises(FileNotFoundError):
        validate_cpu_notebook(tmp_path, SOURCE, **KERNEL)
