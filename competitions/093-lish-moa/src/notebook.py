"""Kaggle 提出用の非公開 CPU Notebook を生成する。"""

import argparse
import json
from base64 import b64encode
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from uuid import uuid4

import numpy as np
import pandas as pd
from model import FILES, TRAIN_FILES, load_training
from train import DATA, selected_run, verify_run

from kumite.provenance import file_sha256

DUMMY = """
from pathlib import Path

import pandas as pd

data = next(path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv')
            if (path.parent / 'test_features.csv').is_file())
test = pd.read_csv(data / 'test_features.csv', usecols=['sig_id', 'cp_type'])
sample = pd.read_csv(data / 'sample_submission.csv')
assert sample['sig_id'].equals(test['sig_id'])
assert test['cp_type'].isin(['trt_cp', 'ctl_vehicle']).all()
sample.iloc[:, 1:] = 0.01
sample.loc[test['cp_type'].eq('ctl_vehicle'), sample.columns[1:]] = 0.0
sample.to_csv('submission.csv', index=False)
"""


def build(output: Path, owner: str, trial: str, source: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "id": "inference",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{owner}/kumite-moa-{trial.lower()}",
        "title": f"Kumite MoA {trial}",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["lish-moa"],
    }
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def verified_weights(
    expected: Path, contract: Mapping[str, object]
) -> tuple[bytes, dict[str, object]]:
    manifest = json.loads(expected.with_suffix(".fit.json").read_text(encoding="utf-8"))
    weights = expected.with_suffix(".npz").read_bytes()
    if (
        manifest["contract"] != contract
        or manifest["weights_sha256"] != sha256(weights).hexdigest()
        or manifest["csv_sha256"] != file_sha256(expected)
    ):
        raise ValueError("提出成果物が生成記録と一致しません")
    with np.load(BytesIO(weights), allow_pickle=False) as state:
        if json.loads(state["contract"].item()) != contract:
            raise ValueError("学習済み係数の学習条件が実験記録と一致しません")
    return weights, manifest


def export(output: Path, owner: str, run_id: str, expected: Path) -> None:
    contract, model_source = verify_run(run_id)
    weights, manifest = verified_weights(expected, contract)
    encoded = "\n".join(repr(chunk) for chunk in wrap(b64encode(weights).decode(), 100))
    hashes = {name: file_sha256(DATA / name) for name in TRAIN_FILES}
    source = f"""
{model_source.decode("utf-8")}

from base64 import b64decode
from hashlib import sha256
from io import BytesIO

root = globals().get('INPUT_ROOT', Path('/kaggle/input'))
output = globals().get('OUTPUT_CSV', Path('submission.csv'))
data = next(path.parent for path in root.rglob('sample_submission.csv')
            if all((path.parent / name).is_file() for name in {FILES!r}))
for name, digest in {hashes!r}.items():
    with (data / name).open('rb') as stream:
        digest_actual = sha256()
        while chunk := stream.read(1 << 20):
            digest_actual.update(chunk)
        if digest_actual.hexdigest() != digest:
            raise ValueError(f'学習データのハッシュ不一致: {{name}}')
payload = (
{encoded}
)
with np.load(BytesIO(b64decode(payload)), allow_pickle=False) as state:
    write_submission(data, dict(state), output)
"""
    output.mkdir(parents=True, exist_ok=True)
    actual = output / f".preview-{uuid4().hex}.csv"
    try:
        exec(compile(source, "submit.ipynb", "exec"), {"INPUT_ROOT": DATA, "OUTPUT_CSV": actual})
        pd.testing.assert_frame_equal(pd.read_csv(actual), pd.read_csv(expected), check_exact=True)
        preview_hash = file_sha256(actual)
    finally:
        actual.unlink(missing_ok=True)
    build(output, owner, run_id, source)
    evidence = {
        "run_id": run_id,
        "training_sha256": hashes,
        "public_input_sha256": {name: file_sha256(DATA / name) for name in FILES[-2:]},
        "public_output_sha256": preview_hash,
        "fit": manifest,
        "notebook_sha256": file_sha256(output / "submit.ipynb"),
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline")
    mode.add_argument("--selection", type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    if args.baseline or args.selection:
        if args.expected is None:
            parser.error("--expected にローカル提出 CSV を指定してください")
        if args.selection:
            _, labels, groups = load_training(DATA)
            run_id = selected_run(args.selection, groups, labels.to_numpy())
        else:
            run_id = args.baseline
            if run_id.partition("-")[0] != "E01":
                parser.error("--baseline には E01 の run ID を指定してください")
        export(args.output, args.owner, run_id, args.expected)
    else:
        build(args.output, args.owner, "E00", DUMMY)
