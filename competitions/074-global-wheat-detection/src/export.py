"""記録済みの CV 設定で全データを学習し、オフライン提出用にまとめる。"""

import argparse
import hashlib
import inspect
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch
from analyze import build_selection
from dataset import ROOT, load_samples
from model import build_model, detect, fit, fuse_views, load_image
from submission import submission_images, write_submission

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import code_version, file_sha256, files_sha256
from kumite.runs import RunStore

IMPORTS = """import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d
from torchvision.transforms import functional as F
"""


def verify_selection(run_id: str, choice: dict[str, object], store: RunStore) -> None:
    rows = choice.get("comparisons")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) or not isinstance(row.get("run_id"), str) for row in rows)
    ):
        raise ValueError("Selection has no candidate records")
    run_ids = [row["run_id"] for row in rows]
    if [store.record(key).params.get("trial") for key in run_ids[:1]] != ["E00"] or sorted(
        store.record(key).params.get("trial", "") for key in run_ids[1:]
    ) != ["E02", "E03", "E04", "E05"]:
        raise ValueError("Selection requires the complete final candidate set")
    expected = build_selection(run_ids[0], run_ids[1:])
    if expected != choice or expected["selected_run"] != run_id:
        raise ValueError("Selection does not match rescored OOF")


def export(
    owner: str,
    cv_run_id: str,
    output: Path,
    selection: Path | None,
    notebook_device: Literal["cpu", "cuda"] = "cuda",
    *,
    run_id: str,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    store = RunStore(ROOT)
    record = store.record(cv_run_id)
    selection_bytes = None
    selection_hash = None
    if record.split_version != "source-v1":
        raise ValueError("Submission requires a source-v1 run")
    if selection is None:
        if record.params.get("trial") != "E00":
            raise ValueError("Final submission requires --selection")
    else:
        selection_bytes = selection.read_bytes()
        verify_selection(cv_run_id, json.loads(selection_bytes), store)
        selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    data = ROOT / "data"
    sample = data / "sample_submission.csv"
    submission_images(sample)
    template_hash = file_sha256(sample)
    samples, _ = load_samples(data)
    if files_sha256([data / "train.csv", *(sample.path for sample in samples)]) != record.data_hash:
        raise ValueError("Training data changed since CV")
    full_samples, _ = load_samples(data, include_unlabeled=True)
    final_data_hash = files_sha256([data / "train.csv", *(sample.path for sample in full_samples)])
    size, nms = int(record.params["size"]), float(record.params["nms"])
    flip_fusion = record.params.get("flip_fusion", False)
    for name in ("oof.jsonl", "split.json"):
        store.artifact_path(cv_run_id, name)
    inputs = store.input_hashes(cv_run_id, "oof.jsonl", "split.json")
    if selection_hash is not None:
        inputs["selection.json"] = selection_hash
    version = code_version(ROOT.parents[1])
    final_record = replace(
        record,
        run_id=run_id,
        data_hash=final_data_hash,
        split_version="full-train",
        split_hash="",
        hypothesis="選定済みの構成を全学習画像で再学習して外部評価する",
        changed_one_thing="全学習画像での再学習と提出物の生成",
        params={
            **record.params,
            "cv_run": cv_run_id,
            "score_source": "source_cv",
            "training_images": len(full_samples),
            "training_device": "cuda",
            "notebook_device": notebook_device,
        },
        code_version=version,
        inputs=inputs,
        artifacts={},
        late_score=None,
        runtime_sec=None,
        peak_memory_mb=None,
        conclusion="",
        next_experiment="",
    )
    dataset = f"{owner}/kumite-wheat-weights-{run_id.lower()}"

    def write_outputs(stage: Path) -> None:
        model = fit(full_samples, size, bool(record.params["color"]), torch.device("cuda"))
        checkpoint = stage / "wheat.pt"
        torch.save(model.state_dict(), checkpoint)
        del model
        write_submission(
            sample,
            checkpoint,
            stage / "public.csv",
            size,
            nms,
            torch.device(notebook_device),
            flip_fusion=flip_fusion,
        )
        weights_hash = file_sha256(checkpoint)
        (stage / "dataset-metadata.json").write_text(
            json.dumps(
                {"title": f"Kumite Wheat {run_id}", "id": dataset, "licenses": [{"name": "other"}]}
            ),
            encoding="utf-8",
        )
        write_notebook(
            stage,
            owner,
            run_id,
            dataset,
            weights_hash,
            template_hash,
            size,
            nms,
            flip_fusion,
            notebook_device,
        )
        if selection_bytes is not None:
            (stage / "selection.json").write_bytes(selection_bytes)
        (stage / "evidence.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "cv_run_id": cv_run_id,
                    "code_version": version,
                    "data_hash": record.data_hash,
                    "final_data_hash": final_data_hash,
                    "final_training_images": len(full_samples),
                    "split_hash": record.split_hash,
                    "weights_hash": weights_hash,
                    "selection_hash": selection_hash,
                    "notebook_hash": file_sha256(stage / "submit.ipynb"),
                    "notebook_device": notebook_device,
                    "sample_submission_sha256": template_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    with store.reserve(run_id) as reservation:
        published = store.publish(reservation, final_record, write_outputs)
    output.mkdir(parents=True, exist_ok=False)
    (output / "weights").mkdir()
    (output / "kernel").mkdir()
    for name, dest in (
        ("wheat.pt", "weights/wheat.pt"),
        ("dataset-metadata.json", "weights/dataset-metadata.json"),
        ("submit.ipynb", "kernel/submit.ipynb"),
        ("kernel-metadata.json", "kernel/kernel-metadata.json"),
        ("public.csv", "public.csv"),
        ("evidence.json", "evidence.json"),
    ):
        shutil.copyfile(published / name, output / dest)


def write_notebook(
    kernel: Path,
    owner: str,
    run_id: str,
    dataset: str,
    weights_hash: str,
    template_hash: str,
    size: int,
    nms: float,
    flip_fusion: bool,
    device: Literal["cpu", "cuda"],
) -> None:
    source = "\n\n".join(
        [
            IMPORTS,
            *(
                inspect.getsource(fn)
                for fn in (
                    build_model,
                    load_image,
                    detect,
                    fuse_views,
                    submission_images,
                    write_submission,
                )
            ),
        ]
    )
    source += f"""
root = Path('/kaggle/input')
sample, = (p for p in root.rglob('sample_submission.csv') if (p.parent / 'test').is_dir())
checkpoint, = root.rglob('wheat.pt')
with checkpoint.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != {weights_hash!r}:
        raise ValueError('Checkpoint hash mismatch')
write_submission(
    sample, checkpoint, Path('/kaggle/working/submission.csv'),
    {size}, {nms}, torch.device({device!r}), flip_fusion={flip_fusion!r},
    hidden_template_sha256={template_hash!r},
)
"""
    compile(source, "wheat-notebook", "exec")
    suffix = "-cpu" if device == "cpu" else ""
    write_cpu_notebook(
        kernel,
        source,
        kernel_id=f"{owner}/kumite-wheat-{run_id.lower()}{suffix}",
        title=f"Kumite Wheat {run_id}{' CPU' if suffix else ''}",
        competition="global-wheat-detection",
    )
    metadata_path = kernel / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(enable_gpu=device == "cuda", dataset_sources=[dataset])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("cv_run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    export(args.owner, args.cv_run_id, args.output, args.selection, args.device, run_id=args.run_id)


if __name__ == "__main__":
    main()
