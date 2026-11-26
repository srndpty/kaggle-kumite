"""選定を再検証して全量学習し、提出 Notebook と重みを生成する。"""

import argparse
import json
from pathlib import Path

import torch
from data import ROOT, dataset
from model import Settings, fit
from train import PLAN, RECEIPT, TRIALS, assess, plan, run_settings, source

from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256
from kumite.quality_gates import publish_export, verify_selection
from kumite.runs import RunStore

SLUG = "sartorius-cell-instance-segmentation"
WEIGHTS = "weights.pt"
NOTEBOOK = "submit.ipynb"
METADATA = "kernel-metadata.json"
REQUIRED = (WEIGHTS, NOTEBOOK, METADATA)

INFERENCE = """\
import csv
from hashlib import file_digest

root = Path("/kaggle/input")
image_dirs = [path / "test" for path in root.rglob(SLUG) if (path / "test").is_dir()]
weight_files = [path for path in root.rglob("weights.pt") if path.parent.name == DATASET]
if len(image_dirs) != 1 or len(weight_files) != 1:
    raise ValueError("公式 test または選定済み重みを一意に特定できません")
with weight_files[0].open("rb") as stream:
    if file_digest(stream, "sha256").hexdigest() != WEIGHT_SHA256:
        raise ValueError("選定済み重みの SHA-256 が一致しません")
paths = sorted(image_dirs[0].glob("*.png"))
ids = [path.stem for path in paths]
if not ids or len(set(ids)) != len(ids):
    raise ValueError("提出画像の ID が不正です")
device = torch.device("cuda")
config = CONFIG
model = build(config, pretrained=False)
model.load_state_dict(torch.load(weight_files[0], map_location="cpu", weights_only=True))
model.to(device)
expected = {}
with Path("submission.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("id", "predicted"))
    for path in paths:
        with PillowImage.open(path) as image:
            size = image.width * image.height
        masks = disjoint(predict(model, path, config, device), size, config["retain"])
        expected[path.stem] = (size, len(masks))
        writer.writerows((path.stem, rle) for rle in (masks or [""]))

with Path("submission.csv").open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    if reader.fieldnames != ["id", "predicted"]:
        raise ValueError("提出ファイルの列が不正です")
    observed = {}
    for row in reader:
        observed.setdefault(row["id"], []).append(row["predicted"])
    if list(observed) != ids:
        raise ValueError("提出ファイルの画像 ID が不正です")
    for image_id, predictions in observed.items():
        size, count = expected[image_id]
        if count == 0 and predictions == [""]:
            continue
        if len(predictions) != count:
            raise ValueError("提出マスク数が不正です")
        ious([], predictions, size)
"""


def notebook_source(config: Settings, dataset_slug: str, digest: str) -> list[str]:
    blocks = []
    for module in ("data", "masks", "model"):
        lines = (ROOT / "src" / f"{module}.py").read_text(encoding="utf-8").splitlines()
        blocks.append(
            "\n".join(
                line
                for line in lines[1:]
                if not line.startswith(("from data import", "from masks import"))
            )
        )
    blocks.insert(0, '__file__ = "/kaggle/working/src/notebook.py"')
    blocks.append(
        f"SLUG = {SLUG!r}\nDATASET = {dataset_slug.split('/')[-1]!r}\n"
        f"CONFIG = {config!r}\nWEIGHT_SHA256 = {digest!r}"
    )
    blocks.append(INFERENCE)
    return blocks


def write_notebook(path: Path, owner: str, dataset_slug: str, config: Settings) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code,
            }
            for code in notebook_source(config, dataset_slug, file_sha256(path / WEIGHTS))
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    metadata = {
        "id": f"{owner}/kumite-077-final",
        "title": "Kumite 077 Final",
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [SLUG],
        "dataset_sources": [dataset_slug],
        "kernel_sources": [],
    }
    for name, value in ((NOTEBOOK, notebook), (METADATA, metadata)):
        (path / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate(path: Path) -> None:
    metadata = json.loads((path / METADATA).read_text(encoding="utf-8"))
    notebook = json.loads((path / NOTEBOOK).read_text(encoding="utf-8"))
    if (
        metadata["enable_internet"]
        or not metadata["is_private"]
        or metadata["competition_sources"] != [SLUG]
        or metadata["code_file"] != NOTEBOOK
        or len(notebook["cells"]) != 6
        or not (path / WEIGHTS).stat().st_size
        or f"WEIGHT_SHA256 = {file_sha256(path / WEIGHTS)!r}" not in notebook["cells"][4]["source"]
    ):
        raise ValueError("提出 Notebook または重みが不正です")
    for cell in notebook["cells"]:
        compile(cell["source"], NOTEBOOK, "exec")


def export(run_id: str, owner: str, dataset_slug: str, path: Path, receipt: Path) -> Path:
    store = RunStore(ROOT)
    selection_plan = plan(path)
    selection = verify_selection(store, selection_plan, receipt, assess)
    winner = store.record(selection.winner)
    trial = next(
        name
        for name in TRIALS
        if (name not in ("S00", "S01") or "S00" in selection_plan.candidates)
        and dict(run_settings(name, path)) == winner.params
    )
    config = run_settings(trial, path)

    def write(directory: Path, record: ExperimentRecord) -> None:
        rows, _, data_hash, split_hash = dataset()
        if (data_hash, split_hash) != (record.data_hash, record.split_hash):
            raise ValueError("最終学習のデータが選定済み CV と一致しません")
        model = fit(rows, config, torch.device("cuda"))
        torch.save(model.cpu().state_dict(), directory / WEIGHTS)
        write_notebook(directory, owner, dataset_slug, config)

    return publish_export(
        store,
        run_id,
        plan=selection_plan,
        receipt=receipt,
        assess=assess,
        source=source(trial, path),
        write=write,
        required_outputs=REQUIRED,
        validate_outputs=validate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("dataset_slug")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    print(export(args.run_id, args.owner, args.dataset_slug, args.plan.resolve(), args.receipt))
