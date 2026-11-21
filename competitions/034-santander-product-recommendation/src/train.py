"""固定した月次 CV で推薦モデルを比較し、Kaggle 提出を生成する。"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from model import DATE, ID, map_at_7, previous_month, rank, transitions, validate_ranking

from kumite.crossfit import CrossFitResult, Split, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CV_MONTHS = (201603, 201604)
HOLDOUT = 201605
TEST = 201606
SEED = 42
VARIANTS = {
    "dummy": ("提出の採点を確認する", "商品列名順"),
    "baseline": ("全期間の商品取得率を基準にする", "基準"),
    "recent": ("直近の需要で改善する", "直近3か月で集計"),
    "portfolio": ("保有商品数で取得率が異なる", "保有数別で集計"),
    "logistic": ("保有商品の組み合わせで改善する", "商品別L2ロジスティック回帰"),
    "seasonal": ("前年同月の取得傾向で改善する", "前年同月の観測の重みを2倍にする"),
}


def load_panel(train: Path, products: list[str]) -> pl.DataFrame:
    return (
        pl.scan_csv(
            train,
            infer_schema=False,
            null_values=["NA", ""],
        )
        .select(
            pl.col(ID).str.strip_chars().cast(pl.Int32),
            pl.col(DATE).str.slice(0, 7).str.replace("-", "").cast(pl.Int32).alias("month"),
            *(
                pl.col(p).str.strip_chars().replace(["", "NA"], None).cast(pl.Int8)
                for p in products
            ),
        )
        .collect(engine="streaming")
    )


def submission_sample(test: Path, sample: Path) -> pl.DataFrame:
    frame = pl.read_csv(sample, schema_overrides={ID: pl.Int32})
    ids = pl.read_csv(test, columns=[ID], schema_overrides={ID: pl.String})[ID]
    ids = ids.str.strip_chars().cast(pl.Int32)
    if (
        frame.columns != [ID, "added_products"]
        or ids.null_count()
        or frame[ID].null_count()
        or ids.n_unique() != len(ids)
        or frame[ID].n_unique() != len(frame)
        or len(ids) != len(frame)
        or not ids.sort().equals(frame[ID].sort())
    ):
        raise ValueError("提出サンプルとテストの顧客IDが不一致です")
    return frame.select(ID)


def write_submission(path: Path, ids: np.ndarray, products: list[str], ranked: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([ID, "added_products"])
        writer.writerows(
            (customer, " ".join(products[int(idx)] for idx in row if idx >= 0))
            for customer, row in zip(ids, ranked, strict=True)
        )


def predict(
    owned: np.ndarray,
    labels: np.ndarray,
    months: np.ndarray,
    query: np.ndarray,
    month: int,
    variant: str,
) -> np.ndarray:
    before = months < month
    if variant == "recent":
        cutoff = previous_month(previous_month(previous_month(month)))
        before &= months >= cutoff
    if variant == "dummy":
        before[:] = False
    train_owned, train_labels = owned[before], labels[before]
    if variant == "seasonal":
        same_month = months == month - 100
        train_owned = np.concatenate((train_owned, owned[same_month]))
        train_labels = np.concatenate((train_labels, labels[same_month]))
    ranked = rank(train_owned, train_labels, query, variant)
    validate_ranking(ranked, query)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    variant: str = args.variant
    store = RunStore(ROOT)
    inputs: dict[str, str] = {}
    selected = None
    if args.final:
        selection = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
        selected = store.record(selection["selected_run_id"])
        if selected.params["variant"] != variant or selection["variant"] != variant:
            parser.error("--final には selection.json で固定した variant が必要です")
        inputs[f"{selected.run_id}/evaluation.npz"] = selected.artifacts["evaluation.npz"]
    train, test, sample = (
        DATA / name for name in ("train_ver2.csv", "test_ver2.csv", "sample_submission.csv")
    )
    with train.open(encoding="utf-8", newline="") as stream:
        products = sorted(
            p for p in next(csv.reader(stream)) if p.endswith("_ult1") and p.startswith("ind_")
        )
    if len(products) != 24:
        raise ValueError("商品列は24列必要です")
    sample_ids = submission_sample(test, sample)
    panel = load_panel(train, products)
    data_hash = files_sha256((train, test, sample))
    if selected is not None and selected.data_hash != data_hash:
        raise ValueError("最終評価のデータがモデル選定時と一致しません")

    with store.reserve(args.run_id) as reservation:
        start = perf_counter()
        blocks = []
        for month in sorted(panel["month"].unique().to_list())[1:]:
            current = panel.filter(pl.col("month") == month).select(ID, *products)
            ids, owned, labels = transitions(panel, current, month, products)
            blocks.append((month, ids, owned, labels))
        test_ids, test_owned, _ = transitions(panel, sample_ids, TEST, products)
        ids = np.concatenate([block[1] for block in blocks])
        owned = np.concatenate([block[2] for block in blocks])
        labels = np.concatenate([block[3] for block in blocks])
        months = np.repeat([block[0] for block in blocks], [len(block[1]) for block in blocks])
        del panel, blocks
        known = (labels >= 0).all(axis=1)
        splits: list[Split] = []
        infos = []
        oof = np.full((len(labels), 7), -1, dtype=np.int8)
        eval_months = (HOLDOUT,) if args.final else CV_MONTHS
        for month in eval_months:
            valid = np.flatnonzero((months == month) & known)
            splits.append((np.flatnonzero(months < month), valid))
            oof[valid] = predict(owned, labels, months, owned[valid], month, variant)
            infos.append(
                {"month": month, "eligible": len(valid), "total": int((months == month).sum())}
            )
        query = predict(owned, labels, months, test_owned, TEST, variant)
        evaluation = Evaluation(
            ROOT.name,
            "map_at_7",
            True,
            map_at_7,
            data_hash,
            "holdout-may-v1" if args.final else "monthly-v1",
            array_sha256(split_roles(len(labels), splits)),
            SEED,
        )
        hypothesis, change = VARIANTS[variant]
        trial = Trial(hypothesis, change, variant, "kaggle-v1", {"variant": variant}, inputs)

        def write_artifacts(out_dir: Path) -> None:
            write_submission(out_dir / "submission.csv", test_ids, products, query)
            valid = np.concatenate([split[1] for split in splits])
            np.savez_compressed(
                out_dir / "evaluation.npz",
                ids=ids[valid],
                labels=labels[valid],
                owned=owned[valid],
                ranked=oof[valid],
                months=months[valid],
                products=products,
            )

        publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            labels,
            splits,
            CrossFitResult(oof, query, infos),
            perf_counter() - start,
            write_artifacts,
        )


if __name__ == "__main__":
    main()
