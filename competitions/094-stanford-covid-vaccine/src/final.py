"""事前に固定した候補から選定し、選定 run の fold 平均 test 予測を提出ファイルとして export する。

選定計画は `selections.toml` に、比較実験の前に書く（候補の優先順と同点許容差）。
選定記録は `outputs/selections/<計画名>.json` に新規作成し、export の入口で再検証する。
"""

import argparse
import json
import tomllib
from pathlib import Path

from data import (
    COMPETITION,
    mcrmse,
    read_train,
    validate_submission,
    write_submission,
)
from run import Method, data_hash, make_splits, source_spec

from kumite.experiments import ExperimentRecord
from kumite.quality_gates import (
    AssessOOF,
    SelectionPlan,
    dense_oof,
    publish_export,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore

PLANS_FILE = Path(__file__).with_name("selections.toml")
SELECTIONS = COMPETITION / "outputs" / "selections"
SUBMISSION = "submission.csv"


def load_plan(name: str, path: Path = PLANS_FILE) -> SelectionPlan:
    plans = tomllib.loads(path.read_text(encoding="utf-8"))
    if name not in plans:
        raise ValueError(f"未定義の選定計画です: {name}")
    entry = plans[name]
    return SelectionPlan(tuple(entry["candidates"]), float(entry["tolerance"]))


def method_of(record: ExperimentRecord) -> Method:
    return Method(**record.params["method"])


def assessor(store: RunStore, plan: SelectionPlan) -> AssessOOF:
    """候補の分割で、正本データから OOF を再採点する関数。候補は同じ分割である前提。"""
    method = method_of(store.record(plan.candidates[0]))
    molecules, labels = read_train()
    y = labels.scored()
    return dense_oof(
        y, make_splits(method, molecules), mcrmse, expected_shape=y.shape, data_hash=data_hash()
    )


def select(name: str) -> None:
    store = RunStore(COMPETITION)
    plan = load_plan(name)
    selection = select_runs(store, plan, assessor(store, plan))
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    write_selection(SELECTIONS / f"{name}.json", selection)
    for run_id, score in selection.scores.items():
        print(f"{run_id}: {score:.6f}")
    print(f"選定: {selection.winner}（許容差 {plan.tolerance}）")


def export(name: str, export_id: str) -> None:
    store = RunStore(COMPETITION)
    plan = load_plan(name)
    receipt = SELECTIONS / f"{name}.json"
    # 勝者のソース設定を組み立てるためだけに読む。記録の正しさは publish_export が再選定で検証する
    winner = store.record(json.loads(receipt.read_text(encoding="utf-8"))["winner"])

    def write(path: Path, selected: ExperimentRecord) -> None:
        query = store.load_array(selected.run_id, "test.npy")
        write_submission(path / SUBMISSION, query)

    published = publish_export(
        store,
        export_id,
        plan=plan,
        receipt=receipt,
        assess=assessor(store, plan),
        source=source_spec(method_of(winner)),
        write=write,
        required_outputs=(SUBMISSION,),
        validate_outputs=lambda path: validate_submission(path / SUBMISSION),
    )
    # export は提出成果物の生成で完了する。Private 評価は analyze.py private で明示的に実行する
    print(f"出力: {published / SUBMISSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("plan")
    e = sub.add_parser("export")
    e.add_argument("plan")
    e.add_argument("export_id")
    args = parser.parse_args()
    if args.command == "select":
        select(args.plan)
    else:
        export(args.plan, args.export_id)
