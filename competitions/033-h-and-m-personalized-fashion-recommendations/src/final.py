"""保存 OOF を再採点して選定を確定し（select）、末尾までの取引で学習した提出を公開する（export）。

export は確定済みの選定記録を照合するだけで、記録がなければ失敗する。
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from data import COMP, FILES, LAST_DAY, Tables, load_tables
from evaluation import K, assessor, validation_windows
from model import METHODS, recommend
from run import load_plan, method_name, source_for

from kumite.experiments import ExperimentRecord
from kumite.provenance import files_sha256
from kumite.quality_gates import (
    AssessOOF,
    publish_export,
    select_runs,
    verify_selection,
    write_selection,
)
from kumite.ranking import validate_full_rankings
from kumite.runs import RunStore, validate_run_id

SUBMISSION = "submission.csv"


def submission_frame(
    ranked: np.ndarray, customer_ids: pl.Series, article_ids: pl.Series
) -> pl.DataFrame:
    """行は提出順の顧客。商品は先頭の 0 を保った ID を空白でつなぐ。"""
    validate_full_rankings(ranked, (len(customer_ids), K), len(article_ids))
    names = article_ids.to_numpy()[ranked.ravel()]
    lists = pl.Series(names, dtype=pl.String).reshape((len(ranked), K)).arr.to_list()
    return pl.DataFrame(
        {"customer_id": customer_ids, "prediction": lists.list.join(" ")},
        schema={"customer_id": pl.String, "prediction": pl.String},
    )


def validate_submission(path: Path, customer_ids: pl.Series, article_ids: pl.Series) -> None:
    """保存したファイルを読み直し、列・顧客の順序・12件の一意な実在商品を確かめる。"""
    frame = pl.read_csv(path, schema_overrides={"customer_id": pl.String, "prediction": pl.String})
    if frame.columns != ["customer_id", "prediction"]:
        raise ValueError("提出の列が不正です")
    if not frame["customer_id"].equals(customer_ids.rename("customer_id")):
        raise ValueError("提出の顧客またはその順序が不正です")
    items = frame["prediction"].str.split(" ")
    if not ((items.list.len() == K) & (items.list.n_unique() == K)).all():
        raise ValueError("提出の件数または重複が不正です")
    if not items.explode().is_in(article_ids.implode()).all():
        raise ValueError("提出に実在しない商品があります")


def receipt_path(selection_id: str) -> Path:
    validate_run_id(selection_id)
    return COMP / "outputs" / "selections" / f"{selection_id}.json"


def load_evaluation() -> tuple[Tables, AssessOOF]:
    tables = load_tables()
    windows = validation_windows(tables.transactions, LAST_DAY)
    return tables, assessor(windows, tables.n_articles, files_sha256(FILES))


def select(plan_path: Path, selection_id: str) -> None:
    """選定を確定して記録する。既存の記録は上書きしない。"""
    _, assess = load_evaluation()
    receipt = receipt_path(selection_id)
    selection = select_runs(RunStore(COMP), load_plan(plan_path), assess)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_selection(receipt, selection)
    print(f"選定 {selection.winner} {selection.scores}", flush=True)


def export(plan_path: Path, selection_id: str, export_id: str) -> Path:
    receipt = receipt_path(selection_id)
    if not receipt.is_file():
        raise FileNotFoundError(f"選定記録 {receipt} がありません。先に select を実行してください")
    plan = load_plan(plan_path)
    tables, assess = load_evaluation()
    store = RunStore(COMP)
    selection = verify_selection(store, plan, receipt, assess)
    name = method_name(selection.winner)

    def write(path: Path, winner: ExperimentRecord) -> None:
        if winner.run_id != selection.winner:
            raise ValueError("選定結果が一致しません")
        customers = np.arange(len(tables.customer_ids), dtype=np.int32)
        result = recommend(tables, METHODS[name], LAST_DAY + 1, customers)
        submission_frame(result.top, tables.customer_ids, tables.article_ids).write_csv(
            path / SUBMISSION
        )
        if result.model is not None:
            (path / "model.txt").write_text(result.model.model_to_string(), encoding="utf-8")

    def validate(path: Path) -> None:
        validate_submission(path / SUBMISSION, tables.customer_ids, tables.article_ids)

    return publish_export(
        store,
        export_id,
        plan=plan,
        receipt=receipt,
        assess=assess,
        source=source_for(name),
        write=write,
        required_outputs=(SUBMISSION,),
        validate_outputs=validate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("plan", type=Path)
        sub.add_argument("selection_id")
        if command == "export":
            sub.add_argument("export_id")
    args = parser.parse_args()
    if args.command == "select":
        select(args.plan, args.selection_id)
    else:
        print(export(args.plan, args.selection_id, args.export_id))
