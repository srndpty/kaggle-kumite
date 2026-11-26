"""F00 の検証済みモデルを使い、入力パス探索だけを変更した F01 を公開する。"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from submission import validate_export
from validation import make_splits, pauc
from workflow import REPO, ROOT, SAMPLE, TRAIN, source

from kumite.provenance import file_sha256
from kumite.quality_gates import SelectionPlan, dense_oof, publish_export, verify_export
from kumite.runs import RunStore

DIRECT_LOOKUP = """    competition = 'isic-2024-challenge'
    dataset = 'kumite-isic-2024-f00-model'
    if name == 'model.txt':
        directories = (root / dataset, root / 'datasets' / 'shion5' / dataset)
    else:
        directories = (root / competition, root / 'competitions' / competition)
    paths = [directory / name for directory in directories if (directory / name).is_file()]
"""


def relocate_notebook(original: Path, destination: Path) -> None:
    payload = json.loads(original.read_text(encoding="utf-8"))
    entry = payload["cells"][-1]
    code = "".join(entry["source"])
    target = "    paths = list(root.rglob(name))\n"
    if code.count(target) != 1:
        raise ValueError("F00 の入力パス探索が想定と一致しません")
    entry["source"] = code.replace(target, DIRECT_LOOKUP).splitlines(keepends=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    plan_path = ROOT / "selection-plan-p6.json"
    config = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = SelectionPlan(tuple(config["candidates"]), config["tolerance"])
    receipt = ROOT / "selection-p6.json"
    frame = pd.read_csv(TRAIN, low_memory=False)
    y = frame["target"].to_numpy(dtype=np.int8)
    assess = dense_oof(
        y, make_splits(frame, y), pauc, expected_shape=y.shape, data_hash=file_sha256(TRAIN)
    )
    sample = pd.read_csv(SAMPLE)
    store = RunStore(ROOT)
    previous = RunStore(ROOT / "outputs/exports")
    prior = verify_export(previous, "F00")
    if prior.params["variant"] != "E04":
        raise ValueError("F00 の選定構成が想定と一致しません")

    def write(path: Path, _) -> None:
        for name in ("model.txt", "sample-predictions.csv"):
            shutil.copyfile(previous.artifact_path("F00", name), path / name)
        relocate_notebook(previous.artifact_path("F00", "submit.ipynb"), path / "submit.ipynb")

    print(
        publish_export(
            store,
            "F01",
            plan=plan,
            receipt=receipt,
            assess=assess,
            source=source("E04", plan_path),
            write=write,
            required_outputs=("model.txt", "sample-predictions.csv", "submit.ipynb"),
            validate_outputs=lambda path: validate_export(path, sample),
            export_files=(Path(__file__).resolve().relative_to(REPO).as_posix(),),
        )
    )


if __name__ == "__main__":
    main()
