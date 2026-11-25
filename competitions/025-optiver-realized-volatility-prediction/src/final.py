"""保存 OOF を再採点して、選定済みモデルの提出 Notebook を生成する。"""

import argparse
import json
from pathlib import Path
from typing import Any

from features import load_rows, predict_submission, rmspe, validate_submission
from run import COMP, DATA, PLAN, data_hash, source_for, splits_for

from kumite.experiments import ExperimentRecord
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.quality_gates import (
    dense_oof,
    publish_export,
    select_runs,
    verify_selection,
    write_selection,
)
from kumite.runs import RunStore

SLUG = COMP.name[4:]
RECEIPT = COMP / "outputs" / "selections" / "P7.json"


def export(owner: str) -> Path:
    rows = load_rows(DATA, "train")
    y = rows["target"].to_numpy()
    splits = splits_for(rows["time_id"].to_numpy())
    assess = dense_oof(y, splits, rmspe, expected_shape=y.shape, data_hash=data_hash())
    store = RunStore(COMP)
    if RECEIPT.exists():
        selection = verify_selection(store, PLAN, RECEIPT, assess)
    else:
        selection = select_runs(store, PLAN, assess)
        write_selection(RECEIPT, selection)
    method = selection.winner.split("-")[0]
    source = Path(__file__).with_name("features.py").read_text(encoding="utf-8")
    source += (
        "\n\ndata, = (path.parent for path in Path('/kaggle/input').rglob('train.csv')"
        " if (path.parent / 'book_train.parquet').is_dir())\n"
        f"predict_submission(data, {method!r}, Path('/kaggle/working/submission.csv'))\n"
    )

    def write(path: Path, winner: ExperimentRecord) -> None:
        if winner.run_id != selection.winner:
            raise ValueError("選定結果が一致しません")
        predict_submission(DATA, method, path / "submission.csv")
        write_cpu_notebook(
            path,
            source,
            kernel_id=f"{owner}/kumite-optiver-final",
            title="Kumite Optiver Final",
            competition=SLUG,
        )

    def validate(path: Path) -> None:
        notebook: dict[str, Any] = json.loads((path / "submit.ipynb").read_text(encoding="utf-8"))
        (cell,) = notebook["cells"]
        if cell["cell_type"] != "code" or "".join(cell["source"]) != source:
            raise ValueError("提出 Notebook のコードが一致しません")
        compile(source, "submit.ipynb", "exec")
        validate_submission(path / "submission.csv", load_rows(DATA, "test")["row_id"])

    return publish_export(
        store,
        "F00",
        plan=PLAN,
        receipt=RECEIPT,
        assess=assess,
        source=source_for(method),
        write=write,
        required_outputs=("submit.ipynb", "kernel-metadata.json", "submission.csv"),
        validate_outputs=validate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    print(export(parser.parse_args().owner))
