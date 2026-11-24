"""保存した3-foldのモデルからローカル予測と提出 Notebook を生成する。"""

import argparse
import base64
import gzip
import json
from pathlib import Path

from pipeline import EVENTS, predict

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SLUG = "child-mind-institute-detect-sleep-states"


def export(run_id: str, owner: str, output: Path) -> None:
    store = RunStore(ROOT)
    record = store.record(run_id)
    models = {
        event: [
            base64.b64encode(
                gzip.compress(
                    store.artifact_path(run_id, f"{event}-{fold}.txt").read_bytes(), mtime=0
                )
            ).decode("ascii")
            for fold in range(3)
        ]
        for event in EVENTS
    }
    options = {
        "long_window": record.params["long_window"],
        "clock": record.params.get("clock", False),
        "suppression": record.params["suppression"],
    }
    output.mkdir(parents=True, exist_ok=True)
    predict(ROOT / "data/test_series.parquet", models, **options).write_csv(
        output / "local-submission.csv"
    )
    source = (Path(__file__).with_name("pipeline.py")).read_text(encoding="utf-8")
    source += f"""
import json

model_path, = Path('/kaggle/input').rglob('sleep-models.json')
models = json.loads(model_path.read_text(encoding='utf-8'))
test_path, = Path('/kaggle/input').rglob('test_series.parquet')
predict(test_path, models, **{options!r}).write_csv('/kaggle/working/submission.csv')
"""
    dataset_id = f"{owner}/kumite-sleep-weights-{run_id.lower()}"
    dataset = output / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "sleep-models.json").write_text(json.dumps(models), encoding="utf-8")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.run_id, args.owner, args.output)
