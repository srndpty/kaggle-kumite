"""Kaggle 採点用の非公開 CPU Notebook を生成する。"""

import argparse
import json
from inspect import getsource
from pathlib import Path

import model
import numpy as np
import pandas as pd
from evidence import MANIFEST, check_manifest, checked_oof, source_hashes
from train import ROOT, SLUG, TRIALS, evaluation_split, read_inputs

from kumite import text_features
from kumite.cv_run import Evaluation
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def competition_data(root: Path) -> Path:
    mounts = [parent / SLUG for parent in (root, root / "competitions")]
    found = [path for path in mounts if path.exists()]
    if len(found) != 1 or not all(
        (found[0] / name).is_file() for name in ("train.csv", "test.csv", "sample_submission.csv")
    ):
        raise ValueError("公式コンペの入力ディレクトリを一意に特定できません")
    return found[0]


def write_submission(data: Path, output: Path, labels: np.ndarray | None = None) -> None:
    template = pd.read_csv(data / "sample_submission.csv", dtype={"qid": str})
    test = pd.read_csv(data / "test.csv", usecols=["qid"], dtype={"qid": str})
    if list(template.columns) != ["qid", "prediction"]:
        raise ValueError("提出テンプレートの列が不正です")
    if (
        test.empty
        or test["qid"].fillna("").str.strip().eq("").any()
        or not test["qid"].is_unique
        or not template["qid"].equals(test["qid"])
    ):
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


def verified_scores(
    store: RunStore,
    evaluation: Evaluation,
    y: np.ndarray,
    sources: dict[str, str],
    trials: tuple[str, ...],
    suffix: str,
) -> tuple[dict[str, float], dict[str, str]]:
    scores = {}
    artifacts = {}
    for trial_id in trials:
        run_id = trial_id + suffix
        record = store.record(run_id)
        check_manifest(store, record, trial_id, TRIALS[trial_id][2], sources)
        _, scores[run_id] = checked_oof(store, record, evaluation, y)
        artifacts.update(store.input_hashes(run_id, "oof.npy", MANIFEST))
    return scores, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=("baseline", "final"))
    parser.add_argument("--run-suffix", default="", help="CV の run ID に付けた共通接尾辞")
    parser.add_argument("--selection", type=Path, help="最終選定記録のパス")
    parser.add_argument("--freeze", action="store_true", help="選定記録を新規作成する")
    args = parser.parse_args()
    if (args.model == "final") != (args.selection is not None):
        parser.error("最終提出には --selection が必要です")
    if args.freeze and args.model != "final":
        parser.error("--freeze は最終提出で使います")
    if args.freeze and args.selection.exists():
        raise FileExistsError(args.selection)
    if args.output.exists():
        raise FileExistsError(args.output)
    source = """import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
"""
    source += f"\nSLUG = {SLUG!r}\n_CHUNK = {1 << 20}\n"
    source += getsource(competition_data)
    source += getsource(write_submission)
    source += """
data = competition_data(Path('/kaggle/input'))
output = Path('/kaggle/working/submission.csv')
"""
    suffix = "p0"
    selection = None
    if args.model is None:
        source += "\nwrite_submission(data, output)\n"
    else:
        sources = source_hashes()
        local_data = ROOT / "data"
        train, _ = read_inputs(local_data)
        y = train["target"].to_numpy()
        evaluation, _ = evaluation_split(local_data, y, model.group_keys(train["question_text"]))
        scores, artifacts = verified_scores(
            RunStore(ROOT),
            evaluation,
            y,
            sources,
            ("E00",) if args.model == "baseline" else tuple(TRIALS),
            args.run_suffix,
        )
        run_id = max(scores, key=scores.__getitem__)
        trial_id = run_id.removesuffix(args.run_suffix)
        suffix = f"{args.model}-{run_id.lower()}"
        kernel_id = f"{args.owner}/kumite-quora-insincere-{suffix}"
        if args.model == "final":
            selection = {
                "run_id": run_id,
                "metric": "F1",
                "split_hash": evaluation.split_hash,
                "data_hash": evaluation.data_hash,
                "candidates": scores,
                "kernel_id": kernel_id,
                "sources": sources,
                "artifacts": artifacts,
            }
            if (
                not args.freeze
                and json.loads(args.selection.read_text(encoding="utf-8")) != selection
            ):
                raise ValueError("確定済みの選定記録と一致しません")
        source += "\n" + getsource(file_sha256)
        train_hash = file_sha256(local_data / "train.csv")
        source += f"\nif file_sha256(data / 'train.csv') != {train_hash!r}:\n"
        source += "    raise ValueError('公式の学習 CSV が CV 入力と一致しません')\n"
        source += "\n" + getsource(text_features) + "\n"
        source += getsource(model).replace("from kumite.text_features import build_features\n", "")
        source += "\n" + getsource(read_inputs)
        source += "\n" + getsource(submit_model).replace("model.", "")
        source += f"\nsubmit_model(data, output, {trial_id == 'E00'}, {TRIALS[trial_id][2]!r})\n"
        if sources != source_hashes():
            raise ValueError("Notebook 生成中に CV ソースが変更されました")
        print(f"提出構成: {run_id}")
    compile(source, "submit.ipynb", "exec")
    if selection is not None and args.freeze:
        with args.selection.open("x", encoding="utf-8") as f:
            f.write(json.dumps(selection, indent=2) + "\n")
    args.output.mkdir(parents=True, exist_ok=False)
    write_cpu_notebook(
        args.output,
        source,
        kernel_id=f"{args.owner}/kumite-quora-insincere-{suffix}",
        title=f"Kumite Quora Insincere {suffix}",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
