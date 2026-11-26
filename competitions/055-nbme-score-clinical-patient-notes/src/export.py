"""選定済みの構成を全量学習し、オフライン推論 Notebook を export する。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch
from data import ROOT
from model import encode, fit, tokenizer
from notebook import SLUG
from train import assess, dataset, plan, source

from kumite.experiments import ExperimentRecord
from kumite.quality_gates import publish_export, verify_selection
from kumite.runs import RunStore

REQUIRED = ("submit.ipynb", "kernel-metadata.json", "model.zip")
INFERENCE = """

(data_dir,) = [p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
               if all((p.parent / name).is_file()
                      for name in ('test.csv', 'patient_notes.csv', 'features.csv'))]
(weights,) = Path('/kaggle/input').rglob('weights.pt')
pairs = load_pairs(data_dir, training=False)
sample_ids = [row['id'] for row in read_rows(data_dir / 'sample_submission.csv')]
pair_by_id = {pair.id: pair for pair in pairs}
if set(sample_ids) != set(pair_by_id) or len(sample_ids) != len(pairs):
    raise ValueError('提出 ID がテストデータと一致しません')
pairs = [pair_by_id[row_id] for row_id in sample_ids]
device = torch.device('cuda')
model_dir = str(weights.parent)
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
model = Tagger(model_dir)
model.load_state_dict(torch.load(weights, weights_only=True))
model.to(device)
scores = predict(model, encode(pairs, tok, trial), len(pairs), tok.pad_token_id, device)

with Path('/kaggle/working/submission.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(('id', 'location'))
    for pair, row in zip(pairs, scores, strict=True):
        spans = decode_pair(pair, row, trial)
        writer.writerow((pair.id, '; '.join(f'{a} {b}' for a, b in spans)))
"""


def write_notebook(path: Path, owner: str, dataset_slug: str, trial: str) -> None:
    cells = []
    for name in ("spans", "data", "model"):
        lines = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8-sig").splitlines()
        code = "\n".join(
            line for line in lines if not line.startswith(("from spans ", "from data "))
        )
        if name == "data":
            code = code.split('if __name__ == "__main__":')[0]
            code = "\n".join(line for line in code.splitlines() if not line.startswith("ROOT = "))
        cells.append((name, code))
    cells.append(("inference", f"trial = {trial!r}\n{INFERENCE}"))
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "id": name,
                "metadata": {},
                "source": code,
                "execution_count": None,
                "outputs": [],
            }
            for name, code in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{owner}/kumite-nbme-final",
        "title": "Kumite NBME Final",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [SLUG],
        "dataset_sources": [dataset_slug],
    }
    for name, value in zip(REQUIRED[:2], (notebook, metadata), strict=True):
        (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(path: Path) -> None:
    notebook = json.loads((path / REQUIRED[0]).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if [cell["id"] for cell in cells] != ["spans", "data", "model", "inference"]:
        msg = "推論 Notebook のセルが一致しません"
        raise ValueError(msg)
    for cell in cells:
        compile(cell["source"], cell["id"], "exec")
    with ZipFile(path / REQUIRED[2]) as archive:
        if not {"weights.pt", "config.json", "tokenizer.json"} <= set(archive.namelist()):
            msg = "オフライン推論に必要なモデルファイルがありません"
            raise ValueError(msg)


def export(run_id: str, owner: str, dataset_slug: str) -> Path:
    store = RunStore(ROOT)
    receipt = ROOT / "selection.json"
    selection = verify_selection(store, plan(), receipt, assess)
    trial = str(store.record(selection.winner).params["trial"])

    def write(path: Path, winner: ExperimentRecord) -> None:
        pairs, _, _, _ = dataset()
        tok = tokenizer()
        pad = tok.pad_token_id
        if not isinstance(pad, int) or winner.params["trial"] != trial:
            msg = "最終学習の設定が選定記録と一致しません"
            raise ValueError(msg)
        model = fit(encode(pairs, tok, trial), pad, torch.device("cuda"))
        with TemporaryDirectory(dir=path.parent) as tmp:
            directory = Path(tmp)
            torch.save(model.cpu().state_dict(), directory / "weights.pt")
            model.encoder.config.save_pretrained(directory)
            tok.save_pretrained(directory)
            with ZipFile(path / REQUIRED[2], "w") as archive:
                for item in directory.iterdir():
                    archive.write(item, item.name)
        write_notebook(path, owner, dataset_slug, trial)

    return publish_export(
        store,
        run_id,
        plan=plan(),
        receipt=receipt,
        assess=assess,
        source=source(trial),
        write=write,
        required_outputs=REQUIRED,
        validate_outputs=validate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("owner")
    parser.add_argument("dataset_slug")
    args = parser.parse_args()
    print(export(args.run_id, args.owner, args.dataset_slug))
