"""検証済みモデルのファイル同一性と提出用改訂を管理する。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile

from data import ROOT

from kumite.provenance import file_sha256
from kumite.quality_gates import EXPORT_EVIDENCE, SELECTION, verify_export
from kumite.runs import RunStore, validate_run_id

NOTEBOOK = "submit.ipynb"
METADATA = "kernel-metadata.json"
MODEL_ARCHIVE = "model.zip"
REQUIRED = (NOTEBOOK, METADATA, MODEL_ARCHIVE)
MODEL_FILES = {"weights.pt", "config.json", "tokenizer.json", "tokenizer_config.json"}
WEIGHT_LOOKUP = "(weights,) = Path('/kaggle/input').rglob('weights.pt')"
VERIFY_MODEL = """
from hashlib import file_digest

for name, expected_hash in model_hashes.items():
    with (weights.parent / name).open('rb') as stream:
        actual_hash = file_digest(stream, 'sha256').hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f'モデルファイルの SHA-256 が一致しません: {name}')
"""


def model_hashes(path: Path) -> dict[str, str]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if (
            not set(names) >= MODEL_FILES
            or len(set(names)) != len(names)
            or any(Path(name).name != name or "/" in name or "\\" in name for name in names)
        ):
            msg = "モデル ZIP に必須ファイルの欠落、重複、または不正なパスがあります"
            raise ValueError(msg)
        hashes: dict[str, str] = {}
        for name in sorted(names):
            with archive.open(name) as stream:
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                hashes[name] = digest.hexdigest()
        return hashes


def model_guard(hashes: dict[str, str]) -> str:
    return f"{WEIGHT_LOOKUP}\nmodel_hashes = {hashes!r}\n{VERIFY_MODEL}"


def guard_inference(code: str, hashes: dict[str, str]) -> str:
    if code.splitlines().count(WEIGHT_LOOKUP) != 1 or VERIFY_MODEL in code:
        msg = "推論 Notebook の重み参照位置が一致しません"
        raise ValueError(msg)
    return code.replace(f"{WEIGHT_LOOKUP}\n", model_guard(hashes) + "\n", 1)


def validate(path: Path) -> None:
    hashes = model_hashes(path / MODEL_ARCHIVE)
    notebook = json.loads((path / NOTEBOOK).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if [cell["id"] for cell in cells] != ["spans", "data", "model", "inference"]:
        msg = "推論 Notebook のセルが一致しません"
        raise ValueError(msg)
    for cell in cells:
        compile(cell["source"], cell["id"], "exec")
    if model_guard(hashes) not in cells[-1]["source"]:
        msg = "Notebook のモデルファイル検証が ZIP と一致しません"
        raise ValueError(msg)


def revise(export_id: str, revision_id: str) -> Path:
    """元の CV と全量学習済み export に結び付けて、ハッシュ検査だけを追加する。"""
    validate_run_id(revision_id)
    store = RunStore(ROOT / "outputs" / "exports")
    record = verify_export(store, export_id)
    selection = json.loads(store.artifact_path(export_id, SELECTION).read_text(encoding="utf-8"))
    winner = selection["winner"]
    archive = store.artifact_path(export_id, MODEL_ARCHIVE)
    hashes = model_hashes(archive)
    notebook = json.loads(store.artifact_path(export_id, NOTEBOOK).read_text(encoding="utf-8"))
    inference = notebook["cells"][-1]
    inference["source"] = guard_inference(inference["source"], hashes)

    root = ROOT / "outputs" / "deployments"
    root.mkdir(parents=True, exist_ok=True)
    output = root / revision_id
    output.mkdir()
    try:
        copyfile(archive, output / MODEL_ARCHIVE)
        copyfile(store.artifact_path(export_id, METADATA), output / METADATA)
        (output / NOTEBOOK).write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        validate(output)
        evidence = {
            "revision_id": revision_id,
            "cv_run_id": winner,
            "cv_evidence_sha256": selection["evidence"][winner],
            "export_id": export_id,
            "export_evidence_sha256": record.artifacts[EXPORT_EVIDENCE],
            "files": {name: file_sha256(output / name) for name in REQUIRED},
            "model_files": hashes,
        }
        (output / "deployment-evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_id")
    parser.add_argument("revision_id")
    args = parser.parse_args()
    print(revise(args.export_id, args.revision_id))
