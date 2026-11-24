"""保存した3-foldのモデルからローカル予測と提出 Notebook を生成する。"""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import lightgbm as lgb
from pipeline import EVENTS, predict
from select_model import validate_main_run
from train import TRIALS

from kumite.experiments import ExperimentRecord
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SLUG = "child-mind-institute-detect-sleep-states"


def validate_final(record: ExperimentRecord, selection_path: Path) -> str:
    payload = selection_path.read_bytes()
    selected = json.loads(payload)
    if record.run_id != selected["selected_run"]:
        raise ValueError("Final export must use the selected run")
    validate_main_run(record, selected["data_hash"], selected["split_hash"])
    return hashlib.sha256(payload).hexdigest()


class InferenceOptions(TypedDict):
    long_window: bool
    clock: bool
    suppression: int


def inference_options(record: ExperimentRecord) -> InferenceOptions:
    params = record.params
    options: InferenceOptions = {
        "long_window": params["long_window"],
        "clock": params.get("clock", False),
        "suppression": params["suppression"],
    }
    if (
        record.model != "LightGBM"
        or type(options["long_window"]) is not bool
        or type(options["clock"]) is not bool
        or type(options["suppression"]) is not int
        or options["suppression"] <= 0
        or type(params["boost_rounds"]) is not int
        or params["boost_rounds"] <= 0
    ):
        raise ValueError(f"{record.run_id}: invalid model parameters")
    return options


def model_bundle(store: RunStore, record: ExperimentRecord, features: int) -> dict[str, list[str]]:
    models: dict[str, list[str]] = {}
    for event in EVENTS:
        models[event] = []
        for fold in range(3):
            payload = store.artifact_path(record.run_id, f"{event}-{fold}.txt").read_bytes()
            source = payload.decode("utf-8")
            model = lgb.Booster(model_str=source)
            if (
                model.num_feature() != features
                or model.current_iteration() != record.params["boost_rounds"]
                or not any(line.startswith("objective=binary ") for line in source.splitlines())
            ):
                raise ValueError(f"{event}-{fold}: model does not match run parameters")
            models[event].append(base64.b64encode(gzip.compress(payload, mtime=0)).decode("ascii"))
    return models


def export(run_id: str, owner: str, output: Path, *, diagnostic: bool = False) -> None:
    store = RunStore(ROOT)
    record = store.record(run_id)
    selection_hash = None if diagnostic else validate_final(record, ROOT / "selection.json")
    options = inference_options(record)
    if not diagnostic:
        _, long_window, radius, suppression = TRIALS[run_id]
        if (
            options
            != {"long_window": long_window, "clock": run_id == "E04", "suppression": suppression}
            or record.params["radius"] != radius
        ):
            raise ValueError("Final model parameters do not match the selected trial")
    source = Path(__file__).with_name("pipeline.py").read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if (
        not diagnostic
        and source_hash != (ROOT / "inference.sha256").read_text(encoding="utf-8").strip()
    ):
        raise ValueError("Inference source SHA-256 mismatch")
    models = model_bundle(store, record, 12 + 10 * options["long_window"] + 2 * options["clock"])
    bundle = json.dumps(models).encode()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    predict(ROOT / "data/test_series.parquet", models, **options).write_csv(
        output / "local-submission.csv"
    )
    source += f"""
model_path, = Path('/kaggle/input').rglob('sleep-models.json')
models = read_model_bundle(model_path, {bundle_hash!r})
test_path, = Path('/kaggle/input').rglob('test_series.parquet')
predict(test_path, models, **{options!r}).write_csv('/kaggle/working/submission.csv')
"""
    dataset_id = f"{owner}/kumite-sleep-weights-{run_id.lower()}"
    dataset = output / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "sleep-models.json").write_bytes(bundle)
    metadata = {
        "id": dataset_id,
        "title": f"Kumite Sleep Weights {run_id}",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    notebook = output / "notebook"
    write_cpu_notebook(
        notebook,
        source,
        kernel_id=f"{owner}/kumite-sleep-{run_id.lower()}",
        title=f"Kumite Sleep {run_id}",
        competition=SLUG,
    )
    metadata_path = notebook / "kernel-metadata.json"
    kernel_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_metadata["dataset_sources"] = [dataset_id]
    metadata_path.write_text(json.dumps(kernel_metadata, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "mode": "diagnostic" if diagnostic else "final",
        "selection_sha256": selection_hash,
        "inference_sha256": source_hash,
        "model_bundle_sha256": bundle_hash,
        "notebook_sha256": file_sha256(notebook / "submit.ipynb"),
        "kernel_metadata_sha256": file_sha256(metadata_path),
        "dataset_metadata_sha256": file_sha256(dataset / "dataset-metadata.json"),
    }
    (output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    export(args.run_id, args.owner, args.output, diagnostic=args.diagnostic)
