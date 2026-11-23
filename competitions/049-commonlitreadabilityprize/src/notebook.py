"""Kaggle の非公開 CPU Notebook を生成する。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from model import load, predict, write_submission
from train import DATA, ROOT, inference_sources, source_signature, verify_selection

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256

SLUG = "commonlitreadabilityprize"

DUMMY = """
from pathlib import Path

import pandas as pd

data = next(path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv')
            if (path.parent / 'train.csv').is_file())
submission = pd.read_csv(data / 'sample_submission.csv')
submission['target'] = pd.read_csv(data / 'train.csv')['target'].mean()
submission.to_csv('submission.csv', index=False)
"""


def build(output: Path, owner: str, trial: str, source: str) -> None:
    title = f"Kumite CommonLit {trial}"
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/{title.lower().replace(' ', '-')}",
        title=title,
        competition=SLUG,
    )


def inference_source(sources: dict[str, str], trial: str, train_sha256: str) -> str:
    features = sources["text_features"]
    model = sources["model"].replace("from kumite.text_features import build_features", "")
    return f"""
from hashlib import sha256

{features}

{model}

def submit(input_root, output):
    data = next(path.parent for path in input_root.rglob('sample_submission.csv')
                if all((path.parent / name).is_file() for name in FILES))
    if sha256((data / 'train.csv').read_bytes()).hexdigest() != {train_sha256!r}:
        raise ValueError('公式学習 CSV が選定時のものと一致しません')
    train, test, sample = load(data)
    pred, _ = predict(train['excerpt'].tolist(), train['target'].to_numpy(),
                      test['excerpt'].tolist(), {trial!r})
    write_submission(test, sample, pred, output)

submit(globals().get('INPUT_ROOT', Path('/kaggle/input')),
       globals().get('OUTPUT_CSV', Path('submission.csv')))
"""


def export_selected(output: Path, owner: str, selection: Path) -> None:
    train, test, sample = load(DATA)
    sources = inference_sources()
    record = verify_selection(selection, train, sources)
    trial = str(record.params["trial"])
    train_hash = file_sha256(DATA / "train.csv")
    source = inference_source(sources, trial, train_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory)
        build(staging, owner, trial, source)
        notebook = json.loads((staging / "submit.ipynb").read_text(encoding="utf-8"))
        actual = staging / "notebook.csv"
        namespace = {"INPUT_ROOT": DATA, "OUTPUT_CSV": actual}
        for cell in notebook["cells"]:
            exec(compile("".join(cell["source"]), "submit.ipynb", "exec"), namespace)
        pred, _ = predict(
            train["excerpt"].tolist(), train["target"].to_numpy(), test["excerpt"].tolist(), trial
        )
        expected = staging / "local.csv"
        write_submission(test, sample, pred, expected)
        pd.testing.assert_frame_equal(
            pd.read_csv(actual, dtype={"id": str}),
            pd.read_csv(expected, dtype={"id": str}),
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        output.mkdir(parents=True, exist_ok=True)
        for name in ("submit.ipynb", "kernel-metadata.json"):
            (output / name).write_bytes((staging / name).read_bytes())
    evidence = {
        "run_id": record.run_id,
        "trial": trial,
        "code_version": record.code_version,
        "data_hash": record.data_hash,
        "split_hash": record.split_hash,
        "training_csv_sha256": train_hash,
        "source_signature": source_signature(sources),
        "notebook_sha256": file_sha256(output / "submit.ipynb"),
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dummy", action="store_true")
    mode.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    if args.dummy:
        build(args.output, args.owner, "E00", DUMMY)
    else:
        export_selected(args.output, args.owner, args.selection)
