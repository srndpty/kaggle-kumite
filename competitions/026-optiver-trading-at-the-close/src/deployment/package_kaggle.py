"""照合済み export を、Kaggle のソース上限に収まる形に梱包する。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.quality_gates import verify_export
from kumite.runs import RunStore

COMP = Path(__file__).resolve().parents[2]


def package(export_id: str) -> Path:
    store = RunStore(COMP / "outputs" / "exports")
    verify_export(store, export_id)
    metadata = json.loads(
        store.artifact_path(export_id, "kernel-metadata.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        store.artifact_path(export_id, "submit.ipynb").read_text(encoding="utf-8")
    )
    (cell,) = notebook["cells"]
    if cell["cell_type"] != "code":
        raise ValueError("未対応の Notebook 形式です")
    payload = "".join(cell["source"]).encode("utf-8")
    digest = sha256(payload).hexdigest()
    kernel_id = metadata["id"]
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
    root = COMP / "outputs" / "deployment" / export_id
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / name).write_bytes(payload)
    dataset_metadata = {
        "id": dataset_id,
        "title": f"{metadata['title']} Payload",
        "licenses": [{"name": "unknown"}],
    }
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    target = root / "notebook"
    (competition,) = metadata["competition_sources"]
    write_cpu_notebook(
        target, source, kernel_id=kernel_id, title=metadata["title"], competition=competition
    )
    metadata["dataset_sources"] = [dataset_id]
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_id")
    print(package(parser.parse_args().export_id))
