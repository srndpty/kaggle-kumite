"""確定した run の設定で全データを学習し、非公開提出用の入力を作る。"""

import argparse
import json
import shutil
from pathlib import Path

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
    args = parser.parse_args()
    store = RunStore(ROOT)
    record = store.record(args.run_id)
    trial = str(record.params["trial"])
    if trial not in TRIALS:
        raise ValueError(f"未知の実験設定: {trial}")
    train = ROOT / "data/train.csv"
    split = ROOT / "outputs/split_v1.csv"
    if (record.data_hash, record.split_hash) != (file_sha256(train), file_sha256(split)):
        raise ValueError("選定した run とデータまたは分割が一致しません")
    rows = load_rows(train, split)
    embeddings = None
    weights = ROOT / "outputs/resnet18-f37072fd.pth"
    if trial in {"E02", "E03", "E04"}:
        embeddings, inputs = load_images(rows, train, ROOT / "outputs/train-embeddings.npz")
        if any(record.inputs.get(key) != value for key, value in inputs.items()) or (
            file_sha256(weights) != inputs["weights_sha256"]
        ):
            raise ValueError("選定した run と画像埋め込みまたは重みが一致しません")
    threshold, model = calibrate(rows, "E03" if trial == "E04" else trial, embeddings)
    encoders = fit_text(rows["title"].tolist(), words=trial == "E01")
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
        "weights_sha256": None if embeddings is None else file_sha256(weights),
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
    (dataset / "model.json").write_text(json.dumps(config), encoding="utf-8")
    for filename in ("matching.py", "images.py", "inference.py"):
        shutil.copy2(ROOT / "src" / filename, dataset)
    if embeddings is not None:
        shutil.copy2(weights, dataset)
    source = f"""import sys
from pathlib import Path

root = Path('/kaggle/input')
bundle, = (path.parent for path in root.rglob('model.json') if path.parent.name == {args.slug!r})
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
