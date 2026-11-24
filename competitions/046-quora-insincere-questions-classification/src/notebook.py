"""Kaggle 採点用の非公開 CPU Notebook を生成する。"""

import argparse
import json
from inspect import getsource
from pathlib import Path

import model
import numpy as np
import pandas as pd
from train import ROOT, TRIALS, read_inputs

from kumite import text_features
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.runs import RunStore

SLUG = "quora-insincere-questions-classification"


def write_submission(data: Path, output: Path, labels: np.ndarray | None = None) -> None:
    template = pd.read_csv(data / "sample_submission.csv", dtype={"qid": str})
    test = pd.read_csv(data / "test.csv", usecols=["qid"], dtype={"qid": str})
    if list(template.columns) != ["qid", "prediction"]:
        raise ValueError("提出テンプレートの列が不正です")
    if not test["qid"].is_unique or not template["qid"].equals(test["qid"]):
        raise ValueError("提出テンプレートとテストの ID が一致しません")
    if labels is not None and (
        labels.shape != (len(template),) or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("提出判定の件数または値が不正です")
    template.assign(prediction=0 if labels is None else labels.astype(int)).to_csv(
        output, index=False
    )


def submit_model(data: Path, output: Path, fixed: bool, options: dict[str, bool]) -> None:
    train, test = read_inputs(data)
    y = train["target"].to_numpy()
    fit, tune = model.splits(y, model.group_keys(train["question_text"]))[0]
    pt, ptest, _ = model.predict(
        train["question_text"].to_numpy(),
        y,
        fit,
        tune,
        test["question_text"].to_numpy(),
        **options,
    )
    for p, n in ((pt, len(tune)), (ptest, len(test))):
        if p.shape != (n,) or not np.all((p >= 0) & (p <= 1)):
            raise ValueError("提出確率の形状または値が不正です")
    threshold = 0.5 if fixed else model.f1_threshold(y[tune], pt)
    print(f"threshold={threshold:.8f}, positives={(ptest >= threshold).sum()}")
    write_submission(data, output, ptest >= threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=("baseline", "final"))
    args = parser.parse_args()
    source = """from pathlib import Path
import numpy as np
import pandas as pd
"""
    source += getsource(write_submission)
    source += """
sample, = (
    path for path in Path('/kaggle/input').rglob('sample_submission.csv')
    if (path.parent / 'test.csv').is_file()
)
output = Path('/kaggle/working/submission.csv')
"""
    suffix = "p0"
    selection = None
    if args.model is None:
        source += "\nwrite_submission(sample.parent, output)\n"
    else:
        if args.model == "baseline":
            run_id = "E00"
        else:
            records = RunStore(ROOT).records()
            candidates = {record.run_id: record for record in records}
            if set(TRIALS) - candidates.keys():
                raise ValueError("最終選定にはすべての試行が必要です")
            evidence = [candidates[key] for key in TRIALS]
            if len({(r.data_hash, r.split_hash, r.metric, r.seed) for r in evidence}) != 1:
                raise ValueError("試行のデータ・分割・指標または seed が一致しません")
            run_id = max(TRIALS, key=lambda key: candidates[key].oof_score)
            selection = {
                "run_id": run_id,
                "metric": "F1",
                "split_hash": candidates[run_id].split_hash,
                "data_hash": candidates[run_id].data_hash,
                "candidates": {r.run_id: r.oof_score for r in evidence},
                "kernel_id": f"{args.owner}/kumite-quora-insincere-final-{run_id.lower()}",
            }
        suffix = f"{args.model}-{run_id.lower()}"
        source += "\n" + getsource(text_features) + "\n"
        source += getsource(model).replace("from kumite.text_features import build_features\n", "")
        source += "\n" + getsource(read_inputs)
        source += "\n" + getsource(submit_model).replace("model.", "")
        source += (
            f"\nsubmit_model(sample.parent, output, {run_id == 'E00'}, {TRIALS[run_id][2]!r})\n"
        )
        print(f"提出構成: {run_id}")
    compile(source, "submit.ipynb", "exec")
    write_cpu_notebook(
        args.output,
        source,
        kernel_id=f"{args.owner}/kumite-quora-insincere-{suffix}",
        title=f"Kumite Quora Insincere {suffix}",
        competition=SLUG,
    )
    if selection is not None:
        (args.output / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
