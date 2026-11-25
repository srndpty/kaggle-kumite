"""保存 OOF を再採点して選定し、末尾までの取引で学習した提出ファイルを公開する。"""

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
from kumite.quality_gates import publish_export, select_runs, verify_selection, write_selection
from kumite.ranking import validate_full_rankings
from kumite.runs import RunStore

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


def export(plan_path: Path, export_id: str) -> Path:
    plan = load_plan(plan_path)
    tables: Tables = load_tables()
    windows = validation_windows(tables.transactions, LAST_DAY)
    assess = assessor(windows, tables.n_articles, files_sha256(FILES))
    store = RunStore(COMP)
    receipt = COMP / "outputs" / "selections" / f"{export_id}.json"
    if receipt.exists():
        selection = verify_selection(store, plan, receipt, assess)
    else:
        selection = select_runs(store, plan, assess)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        write_selection(receipt, selection)
    print(f"選定 {selection.winner} {selection.scores}", flush=True)
    name = method_name(selection.winner)

    def write(path: Path, winner: ExperimentRecord) -> None:
        if winner.run_id != selection.winner:
            raise ValueError("選定結果が一致しません")
        customers = np.arange(len(tables.customer_ids), dtype=np.int32)
        ranked, _, model = recommend(tables, METHODS[name], LAST_DAY + 1, customers)
        submission_frame(ranked, tables.customer_ids, tables.article_ids).write_csv(
            path / SUBMISSION
        )
        if model is not None:
            (path / "model.txt").write_text(model.model_to_string(), encoding="utf-8")

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
    parser.add_argument("plan", type=Path)
    parser.add_argument("export_id")
    args = parser.parse_args()
    print(export(args.plan, args.export_id))
