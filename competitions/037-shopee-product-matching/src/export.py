"""確定した run の設定で全データを学習し、非公開提出用の入力を作る。"""

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path

from export_checks import (
    NOTEBOOK_SOURCE_FILES,
    source_hashes,
    verify_bundle,
    verify_selection,
    verify_sources,
)
from matching import fit_text
from train import ROOT, TRIALS, calibrate, load_images, load_rows

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("slug")
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    store = RunStore(ROOT)
    if not args.diagnostic:
        verify_selection(store, args.run_id, args.selection)
    record = store.record(args.run_id)
    hashes = source_hashes(ROOT / "src")
    verify_sources(record, hashes)
    trial = str(record.params["trial"])
    if trial not in TRIALS:
        raise ValueError(f"未知の実験設定: {trial}")
    train = ROOT / "data/train.csv"
    split = ROOT / "outputs/split_v1.csv"
    if (record.data_hash, record.split_hash) != (file_sha256(train), file_sha256(split)):
        raise ValueError("選定した run とデータまたは分割が一致しません")
    rows = load_rows(train, split)
    embeddings = None
    weights_sha256 = None
    weights = ROOT / "outputs/resnet18-f37072fd.pth"
    if trial in {"E02", "E03", "E04"}:
        embeddings, inputs = load_images(rows, train, ROOT / "outputs/train-embeddings.npz")
        weights_sha256 = inputs["weights_sha256"]
        if any(record.inputs.get(key) != value for key, value in inputs.items()) or (
            file_sha256(weights) != weights_sha256
        ):
            raise ValueError("選定した run と画像埋め込みまたは重みが一致しません")
    threshold, model = calibrate(rows, "E03" if trial == "E04" else trial, embeddings)
    encoders = fit_text(rows["title"].tolist(), words=trial == "E01")
    verify_sources(record, source_hashes(ROOT / "src"))
    dataset = args.output / "dataset"
    dataset.mkdir(parents=True, exist_ok=False)
    dataset_id = f"{args.owner}/{args.slug}"
    (dataset / "dataset-metadata.json").write_text(
        json.dumps({"id": dataset_id, "title": args.slug, "licenses": [{"name": "other"}]}),
        encoding="utf-8",
    )
    config = {
        "trial": trial,
        "run_id": args.run_id,
        "threshold": threshold,
        "model": None if model is None else model.model_to_string(),
        "source_sha256": hashes,
        "weights_sha256": weights_sha256,
        "text": [
            {
                "analyzer": encoder.analyzer,
                "ngram_range": encoder.ngram_range,
                "idf": encoder.idf_.tolist(),
                "vocabulary": encoder.vocabulary_,
            }
            for encoder in encoders
        ],
    }
    config_bytes = json.dumps(config).encode("utf-8")
    (dataset / "model.json").write_bytes(config_bytes)
    for filename in NOTEBOOK_SOURCE_FILES:
        shutil.copy2(ROOT / "src" / filename, dataset)
    files = {name: hashes[name] for name in NOTEBOOK_SOURCE_FILES}
    files["model.json"] = hashlib.sha256(config_bytes).hexdigest()
    if weights_sha256 is not None:
        shutil.copy2(weights, dataset)
        files[weights.name] = weights_sha256
    manifest_bytes = json.dumps(files).encode("utf-8")
    (dataset / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    verify_bundle(dataset, manifest_sha256)
    if args.diagnostic:
        return
    source = f"""import hashlib
import json
import sys
from pathlib import Path

{inspect.getsource(verify_bundle)}

root = Path('/kaggle/input')
bundle, = (path.parent for path in root.rglob('manifest.json') if path.parent.name == {args.slug!r})
verify_bundle(bundle, {manifest_sha256!r})
data, = (path.parent for path in root.rglob('test.csv') if path.parent.name == 'shopee-product-matching')
sys.path.insert(0, str(bundle))
from inference import submission

submission(data, bundle).to_csv('/kaggle/working/submission.csv', index=False)
"""
    notebook = args.output / "notebook"
    write_cpu_notebook(
        notebook,
        source,
        kernel_id=f"{dataset_id}-submission",
        title=f"{args.slug.replace('-', ' ').title()} Submission",
        competition=ROOT.name[4:],
    )
    metadata_path = notebook / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_sources"] = [dataset_id]
    metadata["enable_gpu"] = embeddings is not None
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
