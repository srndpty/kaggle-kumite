"""保存済み CV の実装で全学習行を使う非公開 Notebook を生成する。"""

import argparse
import json
from pathlib import Path

import pandas as pd
from dummy import SLUG
from model import load, predict, write_submission
from train import DATA, PRIMARY_SPLIT, ROOT, evaluation, make_splits, select

from kumite.crossfit import split_roles
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256, files_sha256
from kumite.runs import RunStore


def export(output: Path, owner: str, run_id: str) -> None:
    train, test = load(DATA)
    store = RunStore(ROOT)
    record = store.record(run_id)
    config = evaluation(split_roles(len(train), make_splits(train)), False)
    if (
        record.split_version != PRIMARY_SPLIT
        or record.split_hash != config.split_hash
        or record.data_hash != config.data_hash
    ):
        raise ValueError("提出対象の CV と現在のデータ・分割が一致しません")
    trial = str(record.params["trial"])
    features = store.artifact_path(run_id, "text_features.py")
    model = store.artifact_path(run_id, "model.py")
    if files_sha256([model, features]) != record.params["source_hash"]:
        raise ValueError("保存した推論ソースのハッシュが一致しません")
    source = (
        features.read_text(encoding="utf-8-sig")
        + "\n"
        + model.read_text(encoding="utf-8-sig").replace(
            "from kumite.text_features import build_features", ""
        )
    )
    source += f"""
input_root = globals().get('INPUT_ROOT', Path('/kaggle/input'))
output_csv = globals().get('OUTPUT_CSV', Path('/kaggle/working/submission.csv'))
data, = (
    path.parent for path in input_root.rglob('summaries_train.csv')
    if all((path.parent / name).is_file() for name in FILES)
)
from hashlib import sha256

for name, expected in { {name: file_sha256(DATA / name) for name in ("summaries_train.csv", "prompts_train.csv")}!r}.items():
    if sha256((data / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'学習データが一致しません: {{name}}')
train, test = load(data)
write_submission(test, predict(train, test, {trial!r}), output_csv, data / 'sample_submission.csv')
"""
    title = f"Kumite Summaries {run_id}"
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/{title.lower().replace(' ', '-')}",
        title=title,
        competition=SLUG,
    )
    notebook = json.loads((output / "submit.ipynb").read_text(encoding="utf-8"))
    actual = output / "notebook.csv"
    expected = output / "local.csv"
    namespace = {"INPUT_ROOT": DATA, "OUTPUT_CSV": actual}
    exec(compile("".join(notebook["cells"][0]["source"]), "submit.ipynb", "exec"), namespace)
    write_submission(test, predict(train, test, trial), expected, DATA / "sample_submission.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(actual, dtype={"student_id": str}),
        pd.read_csv(expected, dtype={"student_id": str}),
        rtol=0,
        atol=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", nargs="?", const="E00", metavar="RUN_ID")
    mode.add_argument("--selection", type=Path)
    args = parser.parse_args()
    run_id = args.baseline
    if args.selection:
        train, _ = load(DATA)
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection != select(train, list(selection["comparisons"])):
            raise ValueError("選定記録と保存済みの OOF が一致しません")
        run_id = selection["selected_run"]
    if not isinstance(run_id, str) or (args.baseline and run_id.partition("-")[0] != "E00"):
        raise ValueError("提出対象の run ID が不正です")
    export(args.output, args.owner, run_id)


if __name__ == "__main__":
    main()
