"""OOF から選定を確定し、別操作で Kaggle Notebook を出力する。"""

import argparse
import json
from pathlib import Path

from train import COMP, TRIALS, load_cv, mcrmse, read_plan, source_spec

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.quality_gates import (
    dense_oof,
    publish_export,
    select_runs,
    verify_selection,
    write_selection,
)
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "export"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--owner")
    parser.add_argument("--export-id")
    args = parser.parse_args()
    store = RunStore(COMP)
    plan_file = args.plan.resolve()
    plan = read_plan(plan_file)
    _, labels, _, splits, evaluation = load_cv()
    assess = dense_oof(
        labels, splits, mcrmse, expected_shape=labels.shape, data_hash=evaluation.data_hash
    )
    if args.action == "select":
        selection = select_runs(store, plan, assess)
        write_selection(args.receipt, selection)
        print(selection)
        return
    if not args.owner or not args.export_id:
        parser.error("export requires --owner and --export-id")
    selection = verify_selection(store, plan, args.receipt, assess)
    block = TRIALS[selection.winner.split("-")[0]][0]
    code = (COMP / "src" / "model.py").read_text(encoding="utf-8")
    code += (
        "\nnp.seterr(invalid='raise')\n"
        "root, = (p.parent for p in Path('/kaggle/input').rglob('train.csv') "
        "if p.parent.name == SLUG)\n"
        f"write_submission(root, Path('/kaggle/working/submission.csv'), {block!r})\n"
    )
    kernel = f"{args.owner}/kumite-ell-{args.export_id.lower()}"
    slug = "feedback-prize-english-language-learning"

    def write(output: Path, _winner: object) -> None:
        write_cpu_notebook(
            output, code, kernel_id=kernel, title=kernel.split("/")[1], competition=slug
        )

    def validate(output: Path) -> None:
        notebook = json.loads((output / "submit.ipynb").read_text(encoding="utf-8"))
        metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
        actual_code = "".join(notebook["cells"][0]["source"])
        if actual_code != code or metadata["competition_sources"] != [slug]:
            raise ValueError("Notebook のコードまたはコンペ参照が不正です")
        compile(actual_code, "submit.ipynb", "exec")

    print(
        publish_export(
            store,
            args.export_id,
            plan=plan,
            receipt=args.receipt,
            assess=assess,
            source=source_spec(block, plan_file),
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
