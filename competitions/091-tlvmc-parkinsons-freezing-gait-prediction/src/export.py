"""固定済み run のモデルと特徴量処理を Kaggle Notebook にまとめる。"""

import argparse
import hashlib
import json
import zlib
from pathlib import Path

from pipeline import EVENTS
from prepare import ROOT

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.runs import RunStore

SLUG = "tlvmc-parkinsons-freezing-gait-prediction"


def export(run_id: str, owner: str, output: Path) -> None:
    store = RunStore(ROOT)
    params = store.record(run_id).params
    trial = params.get("feature_trial", params["trial"])
    model_bytes = store.artifact_path(run_id, "models.json").read_bytes()
    models = json.loads(model_bytes)
    if not models or any(len(values) != len(EVENTS) for values in models.values()):
        raise ValueError("Unexpected model artifact")
    bundle = zlib.compress(model_bytes)
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    kernel = run_id.lower().replace("_", "-")
    dataset_id = f"{owner}/kumite-fog-weights-{kernel}"
    dataset = output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "fog-models.json.zlib").write_bytes(bundle)
    metadata = {
        "id": dataset_id,
        "title": f"Kumite FOG Weights {run_id}",
        "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pipeline = Path(__file__).with_name("pipeline.py").read_text(encoding="utf-8")
    source = f"""{pipeline}

import hashlib
import json
import zlib

model_path, = Path('/kaggle/input').rglob('fog-models.json.zlib')
bundle = model_path.read_bytes()
if hashlib.sha256(bundle).hexdigest() != {bundle_hash!r}:
    raise ValueError('Unexpected model bundle hash')
models = json.loads(zlib.decompress(bundle))
test_dir, = (path for path in Path('/kaggle/input').rglob('test')
             if (path / 'tdcsfog').is_dir() or (path / 'defog').is_dir())
ids, predictions = predict_recordings(recordings(test_dir.parent, 'test'), {trial!r}, models)
submission_frame(ids, predictions).to_csv('/kaggle/working/submission.csv', index=False)
"""
    notebook = output / "notebook"
    write_cpu_notebook(
        notebook,
        source,
        kernel_id=f"{owner}/kumite-fog-{kernel}",
        title=f"Kumite FOG {run_id}",
        competition=SLUG,
    )
    metadata_path = notebook / "kernel-metadata.json"
    kernel_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_metadata["dataset_sources"] = [dataset_id]
    metadata_path.write_text(json.dumps(kernel_metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.run_id, args.owner, args.output)


if __name__ == "__main__":
    main()
