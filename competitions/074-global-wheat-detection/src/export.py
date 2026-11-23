"""記録済みの CV 設定で全データを学習し、オフライン提出用にまとめる。"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from dataset import ROOT, load_samples
from model import build_model, detect, fit, fuse_views, load_image
from submission import write_submission

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256, files_sha256
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


def export(owner: str, run_id: str, output: Path, selection: Path | None) -> None:
    record = RunStore(ROOT).record(run_id)
    if record.split_version != "source-v1":
        raise ValueError("Submission requires a source-v1 run")
    if selection is None:
        if record.params.get("trial") != "E00":
            raise ValueError("Final submission requires --selection")
    else:
        choice = json.loads(selection.read_text(encoding="utf-8"))
        if (
            choice["selected_run"] != run_id
            or choice["data_hash"] != record.data_hash
            or choice["split_hash"] != record.split_hash
        ):
            raise ValueError("Selection does not match the requested run")
    data = ROOT / "data"
    samples, _ = load_samples(data)
    if files_sha256([data / "train.csv", *(sample.path for sample in samples)]) != record.data_hash:
        raise ValueError("Training data changed since CV")
    full_samples, _ = load_samples(data, include_unlabeled=True)
    final_data_hash = files_sha256([data / "train.csv", *(sample.path for sample in full_samples)])
    size, nms = int(record.params["size"]), float(record.params["nms"])
    flip_fusion = record.params.get("flip_fusion", False)
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    device = torch.device("cuda")
    model = fit(full_samples, size, bool(record.params["color"]), device)
    checkpoint = weights / "wheat.pt"
    torch.save(model.state_dict(), checkpoint)
    write_submission(
        data / "test",
        checkpoint,
        output / "public.csv",
        size,
        nms,
        device,
        flip_fusion=flip_fusion,
    )
    dataset = f"{owner}/kumite-wheat-weights-{run_id.lower()}"
    weights_hash = file_sha256(checkpoint)
    (weights / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": f"Kumite Wheat {run_id}", "id": dataset, "licenses": [{"name": "other"}]}
        ),
        encoding="utf-8",
    )
    source = "\n\n".join(
        [
            IMPORTS,
            *(
                inspect.getsource(fn)
                for fn in (build_model, load_image, detect, fuse_views, write_submission)
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
    sample.parent / 'test', checkpoint, Path('/kaggle/working/submission.csv'),
    {size}, {nms}, torch.device('cuda'), flip_fusion={flip_fusion!r},
)
"""
    compile(source, "wheat-notebook", "exec")
    kernel = output / "kernel"
    write_cpu_notebook(
        kernel,
        source,
        kernel_id=f"{owner}/kumite-wheat-{run_id.lower()}",
        title=f"Kumite Wheat {run_id}",
        competition="global-wheat-detection",
    )
    metadata_path = kernel / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(enable_gpu=True, dataset_sources=[dataset])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "evidence.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "data_hash": record.data_hash,
                "final_data_hash": final_data_hash,
                "final_training_images": len(full_samples),
                "split_hash": record.split_hash,
                "weights_hash": weights_hash,
                "selection_hash": file_sha256(selection) if selection is not None else None,
                "notebook_hash": file_sha256(kernel / "submit.ipynb"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    export(args.owner, args.run_id, args.output, args.selection)


if __name__ == "__main__":
    main()
