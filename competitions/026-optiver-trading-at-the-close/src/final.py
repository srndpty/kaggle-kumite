"""保存 OOF を再採点し、選定済みモデルの逐次推論 Notebook を出力する。"""

import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from evaluation import mae, splits_for
from features import Replay, feature_matrix, fit_model, prediction_frame
from p0 import write_notebook
from run import COMP, DATA, load_plan, load_rows, source_for

from kumite.experiments import ExperimentRecord
from kumite.provenance import files_sha256
from kumite.quality_gates import (
    dense_oof,
    publish_export,
    select_runs,
    verify_selection,
    write_selection,
)
from kumite.runs import RunStore


def notebook_source(model: str, method: str) -> str:
    encoded = base64.b64encode(zlib.compress(model.encode())).decode()
    features = Path(__file__).with_name("features.py").read_text(encoding="utf-8")
    return (
        features
        + f"""

import base64
import zlib
import optiver2023

model = lgb.Booster(model_str=zlib.decompress(base64.b64decode({encoded!r})).decode())
replay = Replay({method!r})
env = optiver2023.make_env()
for test, _revealed_targets, sample in env.iter_test():
    values = np.asarray(model.predict(replay.transform(test), num_threads=1))
    env.predict(prediction_frame(test, sample, values))
"""
    )


def export(owner: str, plan_path: Path, export_id: str) -> Path:
    plan = load_plan(plan_path)
    rows = load_rows()
    y = rows["target"].to_numpy()
    splits = splits_for(rows["date_id"].to_numpy(), y)
    assess = dense_oof(y, splits, mae, expected_shape=y.shape, data_hash=files_sha256([DATA]))
    store = RunStore(COMP)
    receipt = COMP / "outputs" / "selections" / f"{export_id}.json"
    if receipt.exists():
        selection = verify_selection(store, plan, receipt, assess)
    else:
        selection = select_runs(store, plan, assess)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        write_selection(receipt, selection)
    method = selection.winner.split("-")[0]

    def write(path: Path, winner: ExperimentRecord) -> None:
        if winner.run_id != selection.winner:
            raise ValueError("選定結果が一致しません")
        labeled = np.isfinite(y)
        model = fit_model(feature_matrix(rows, method).loc[labeled], y[labeled])
        model_text = model.model_to_string()
        (path / "model.txt").write_text(model_text, encoding="utf-8")
        write_notebook(path, owner, notebook_source(model_text, method), stage=export_id.lower())

        example_dir = DATA.parent / "example_test_files"
        example = pd.read_csv(example_dir / "test.csv")
        sample = pd.read_csv(example_dir / "sample_submission.csv", dtype={"row_id": "string"})
        replay = Replay(method)
        for (_, batch), (_, template) in zip(
            example.groupby("time_id", sort=True),
            sample.groupby("time_id", sort=True),
            strict=True,
        ):
            prediction = np.asarray(model.predict(replay.transform(batch), num_threads=1))
            prediction_frame(batch, template[["row_id", "target"]], prediction)

    def validate(path: Path) -> None:
        notebook = json.loads((path / "submit.ipynb").read_text(encoding="utf-8"))
        (cell,) = notebook["cells"]
        source = "".join(cell["source"])
        model = (path / "model.txt").read_text(encoding="utf-8")
        if cell["cell_type"] != "code" or source != notebook_source(model, method):
            raise ValueError("提出 Notebook のコードが一致しません")
        compile(source, "submit.ipynb", "exec")

    return publish_export(
        store,
        export_id,
        plan=plan,
        receipt=receipt,
        assess=assess,
        source=source_for(method),
        write=write,
        required_outputs=("model.txt", "submit.ipynb", "kernel-metadata.json"),
        validate_outputs=validate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--export-id", default="F00")
    args = parser.parse_args()
    print(export(args.owner, args.plan, args.export_id))
