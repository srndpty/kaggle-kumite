"""検証済み export を Kaggle の Dataset と Notebook 用に配置する。"""

import argparse
import json
from pathlib import Path
from shutil import copyfile

from data import ROOT
from export import METADATA, NOTEBOOK, WEIGHTS

from kumite.provenance import file_sha256
from kumite.quality_gates import EXPORT_EVIDENCE, verify_export
from kumite.runs import RunStore


def stage(export_id: str, output: Path) -> None:
    store = RunStore(ROOT / "outputs" / "exports")
    record = verify_export(store, export_id)
    metadata_file = store.artifact_path(export_id, METADATA)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    (dataset_id,) = metadata["dataset_sources"]
    dataset_metadata = {
        "id": dataset_id,
        "title": f"Kumite 077 {export_id} weights",
        "licenses": [{"name": "other"}],
    }
    output.mkdir(parents=True)
    dataset_dir = output / "dataset"
    dataset_dir.mkdir()
    copyfile(store.artifact_path(export_id, WEIGHTS), dataset_dir / WEIGHTS)
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8"
    )
    kernel_dir = output / "kernel"
    kernel_dir.mkdir()
    copyfile(metadata_file, kernel_dir / METADATA)
    copyfile(store.artifact_path(export_id, NOTEBOOK), kernel_dir / NOTEBOOK)
    evidence = {
        "export_id": export_id,
        "export_evidence_sha256": record.artifacts[EXPORT_EVIDENCE],
        "deployment_source_sha256": file_sha256(Path(__file__)),
    }
    (output / "deployment-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.export_id, args.output)
