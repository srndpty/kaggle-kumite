"""選定記録を照合して非公開 CPU Notebook を生成する。"""

import argparse
from pathlib import Path

from model import SLUG
from train import (
    COMP,
    TRIALS,
    assess_for,
    check_candidates,
    load_cv,
    read_plan,
    source_spec,
    trial_for,
)

from kumite.kaggle_notebook import validate_cpu_notebook, write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.quality_gates import publish_export, verify_selection
from kumite.runs import RunStore


def notebook_source(block: str, input_root: Path, submission: Path) -> str:
    source = (COMP / "src/model.py").read_text(encoding="utf-8-sig")
    return source + (
        f"\nroot, = (p.parent for p in Path({input_root.as_posix()!r}).rglob('train.csv') "
        "if p.parent.name == SLUG)\n"
        f"write_submission(root, Path({submission.as_posix()!r}), {block!r})\n"
    )


def write_notebook(
    output: Path,
    block: str,
    kernel: str,
    *,
    input_root: Path = Path("/kaggle/input"),
    submission: Path = Path("/kaggle/working/submission.csv"),
) -> str:
    code = notebook_source(block, input_root, submission)
    write_cpu_notebook(output, code, kernel_id=kernel, title=kernel.split("/")[1], competition=SLUG)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--export-id", required=True)
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = read_plan(path)
    store = RunStore(COMP)
    check_candidates(store, plan)
    _, labels, _, splits, evaluation = load_cv()
    assess = assess_for(labels, splits, evaluation)
    selection = verify_selection(store, plan, args.receipt, assess)
    block = TRIALS[trial_for(selection.winner)][0]
    kernel = f"{args.owner}/kumite-effective-{args.export_id.lower()}"
    code = notebook_source(block, Path("/kaggle/input"), Path("/kaggle/working/submission.csv"))

    def write(output: Path, _winner: object) -> None:
        write_notebook(output, block, kernel)

    def validate(output: Path) -> None:
        validate_cpu_notebook(
            output, code, kernel_id=kernel, title=kernel.split("/")[1], competition=SLUG
        )

    print(
        publish_export(
            store,
            args.export_id,
            plan=plan,
            receipt=args.receipt,
            assess=assess,
            source=source_spec(block, path),
            write=write,
            required_outputs=("submit.ipynb", "kernel-metadata.json"),
            validate_outputs=validate,
            export_files=(
                Path(__file__).relative_to(REPO_ROOT).as_posix(),
                "src/kumite/kaggle_notebook.py",
            ),
        )
    )


if __name__ == "__main__":
    main()
