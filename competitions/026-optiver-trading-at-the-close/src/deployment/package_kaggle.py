"""照合済み export を、Kaggle のソース上限に収まる形に梱包する。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import cast

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.quality_gates import verify_export
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "deployment-manifest.json"


def deployment_manifest(
    export_id: str, payload: bytes, loader_source: str, dataset_id: str
) -> dict[str, str]:
    """export と配布物の対応を、後から監査できる形にする。"""
    return {
        "export_id": export_id,
        "source_artifact": "submit.ipynb",
        "payload_sha256": sha256(payload).hexdigest(),
        "loader_sha256": sha256(loader_source.encode()).hexdigest(),
        "dataset_id": dataset_id,
    }


def _reject_replacement(path: Path, expected: object) -> None:
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != expected:
        raise FileExistsError("確定済みの配布物は置換できません")


def _loader_text(path: Path) -> str:
    notebook = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    cells = cast(list[dict[str, object]], notebook["cells"])
    (cell,) = cells
    return "".join(cast(list[str], cell["source"]))


def commit_deployment(
    root: Path,
    *,
    manifest: dict[str, str],
    payload_name: str,
    payload: bytes,
    dataset_metadata: dict[str, object],
    loader_source: str,
    kernel_id: str,
    title: str,
    competition: str,
    kernel_metadata: dict[str, object],
) -> None:
    """同一内容の再生成だけを許し、確定済み配布物の置換を拒否する。"""
    dataset = root / "dataset"
    notebook = root / "notebook"
    if dataset.is_dir():
        payloads = list(dataset.glob("optiver-close-*.py.txt"))
        if any(path.name != payload_name for path in payloads):
            raise FileExistsError("確定済みの配布物は置換できません")
        expected_payload = dataset / payload_name
        if expected_payload.is_file() and expected_payload.read_bytes() != payload:
            raise FileExistsError("確定済みの配布物は置換できません")
    _reject_replacement(dataset / "dataset-metadata.json", dataset_metadata)
    loader = notebook / "submit.ipynb"
    if loader.is_file() and _loader_text(loader) != loader_source:
        raise FileExistsError("確定済みの配布物は置換できません")
    _reject_replacement(notebook / "kernel-metadata.json", kernel_metadata)
    recorded = root / MANIFEST_NAME
    if recorded.is_file() and json.loads(recorded.read_text(encoding="utf-8")) != manifest:
        raise FileExistsError("確定済みの配布物は置換できません")

    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / payload_name).write_bytes(payload)
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    write_cpu_notebook(
        notebook, loader_source, kernel_id=kernel_id, title=title, competition=competition
    )
    (notebook / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2), encoding="utf-8"
    )
    recorded.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def package(export_id: str) -> Path:
    store = RunStore(COMP / "outputs" / "exports")
    verify_export(store, export_id)
    metadata_path = store.artifact_path(export_id, "kernel-metadata.json")
    metadata = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
    notebook_path = store.artifact_path(export_id, "submit.ipynb")
    notebook = cast(dict[str, object], json.loads(notebook_path.read_text(encoding="utf-8")))
    cells = cast(list[dict[str, object]], notebook["cells"])
    (cell,) = cells
    if cell["cell_type"] != "code":
        raise ValueError("未対応の Notebook 形式です")
    payload = "".join(cast(list[str], cell["source"])).encode()
    digest = sha256(payload).hexdigest()
    kernel_id = cast(str, metadata["id"])
    dataset_id = f"{kernel_id}-payload"
    name = f"optiver-close-{digest[:16]}.py.txt"
    source = f"""from hashlib import sha256
from pathlib import Path

payload_path, = Path('/kaggle/input').rglob({name!r})
payload = payload_path.read_bytes()
if sha256(payload).hexdigest() != {digest!r}:
    raise ValueError('Export SHA-256 mismatch')
exec(compile(payload, str(payload_path), 'exec'), {{'__name__': '__main__'}})
"""
    compile(source, "submit.ipynb", "exec")
    manifest = deployment_manifest(export_id, payload, source, dataset_id)
    dataset_metadata: dict[str, object] = {
        "id": dataset_id,
        "title": f"{metadata['title']} Payload",
        "licenses": [{"name": "unknown"}],
    }
    competitions = cast(list[str], metadata["competition_sources"])
    (competition,) = competitions
    metadata["dataset_sources"] = [dataset_id]
    title = cast(str, metadata["title"])
    commit_deployment(
        COMP / "outputs" / "deployment" / export_id,
        manifest=manifest,
        payload_name=name,
        payload=payload,
        dataset_metadata=dataset_metadata,
        loader_source=source,
        kernel_id=kernel_id,
        title=title,
        competition=competition,
        kernel_metadata=metadata,
    )
    return COMP / "outputs" / "deployment" / export_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_id")
    print(package(parser.parse_args().export_id))
