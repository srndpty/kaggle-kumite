"""固定分割、実験、選定と最終推論。"""

import argparse
import json
from hashlib import sha256
from math import isclose
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import (
    FILES,
    HYPOTHESES,
    SEED,
    load_training,
    log_loss,
    losses,
    predict,
    treated,
)
from sklearn.model_selection import GroupKFold, KFold

from kumite.checks import validate_multilabel_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, file_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = Path(__file__).with_name("model.py")
VERSION = "moa-drug-group-v1"
ROW_VERSION = "moa-row-v1"


def make_folds(groups: np.ndarray, *, shuffle_rows: bool = False) -> np.ndarray:
    folds = np.full(len(groups), -1)
    x = np.empty(len(groups))
    splits = (
        KFold(5, shuffle=True, random_state=SEED).split(x)
        if shuffle_rows
        else GroupKFold(5).split(x, groups=groups)
    )
    for fold, (training, validation) in enumerate(splits):
        if not shuffle_rows and np.intersect1d(groups[training], groups[validation]).size:
            raise ValueError("化合物が fold をまたいでいます")
        if (folds[validation] != -1).any():
            raise ValueError("同じ行が複数回検証されます")
        folds[validation] = fold
    if (folds < 0).any():
        raise ValueError("検証対象外の行があります")
    return folds


def evaluation(folds: np.ndarray, row: bool = False) -> Evaluation:
    return Evaluation(
        "lish-moa",
        "mean_columnwise_log_loss",
        False,
        log_loss,
        files_sha256(DATA / name for name in FILES),
        ROW_VERSION if row else VERSION,
        array_sha256(folds),
        SEED,
    )


def prepare(train: pd.DataFrame, labels: pd.DataFrame, groups: np.ndarray) -> None:
    folds = make_folds(groups)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"sig_id": train.index, "fold": folds}).to_csv(output / "folds.csv", index=False)
    group_labels = labels.groupby(groups).max()
    summary = pd.DataFrame({"positive_rows": labels.sum(), "positive_drugs": group_labels.sum()})
    summary.to_csv(output / "label_counts.csv")
    print(
        f"rows={len(train)} treated={int(treated(train).sum())} "
        f"labels={len(labels.columns)} drugs={len(group_labels)}"
    )
    print("drug sizes:", pd.Series(groups).value_counts().describe().to_dict())
    print(summary.describe().to_string())
    print("fold rows:", np.bincount(folds).tolist())


def cv(
    store: RunStore,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    groups: np.ndarray,
    trial: str,
    run_id: str,
    row: bool,
    baseline: str | None,
) -> None:
    if run_id.partition("-")[0] != trial or (row and trial != "E01"):
        raise ValueError("実験番号または行分割の指定が不正です")
    folds = make_folds(groups, shuffle_rows=row)
    y = labels.to_numpy()
    runspec = evaluation(folds, row)
    splits = folds_to_splits(folds)
    inputs: dict[str, str] = {}
    baseline_oof = None
    if baseline is not None:
        record = store.record(baseline)
        if (
            trial != "E03"
            or record.params["trial"] != "E01"
            or (record.data_hash, record.split_hash, record.split_version)
            != (runspec.data_hash, runspec.split_hash, runspec.split_version)
        ):
            raise ValueError("E03 の比較元は同じ分割の E01 を指定してください")
        baseline_oof = store.load_array(baseline, "oof.npy")
        validate_multilabel_probabilities(baseline_oof, *y.shape)
        inputs[f"{baseline}/oof.npy"] = record.artifacts["oof.npy"]
    model_name = {"E00": "定数確率", "E04": "MLP"}.get(trial, "L2 LogisticRegression")
    hypothesis = "P0 と同じ処理群 0.01・対照群 0" if trial == "E00" else HYPOTHESES[trial]
    change = "化合物単位から行シャッフルへ変更" if row else hypothesis

    def fit(training: np.ndarray, validation: np.ndarray) -> FoldOutput:
        if trial == "E00":
            probability = 0.01 * treated(train.iloc[validation])[:, None]
            prediction = np.broadcast_to(probability, (len(validation), y.shape[1]))
        elif baseline_oof is None:
            prediction = predict(train.iloc[training], y[training], train.iloc[validation], trial)
        else:
            prediction = baseline_oof[validation].copy()
            selected = treated(train.iloc[validation])
            label = y[training][treated(train.iloc[training])]
            prediction[selected] = 0.9 * prediction[selected] + 0.1 * (label.sum(axis=0) + 0.5) / (
                len(label) + 1
            )
        return FoldOutput(
            prediction,
            np.empty(0),
            {
                "zero_positive_labels": int((y[validation].sum(axis=0) == 0).sum()),
                "shared_drugs": int(np.intersect1d(groups[training], groups[validation]).size),
            },
        )

    def write(output: Path) -> None:
        pd.DataFrame({"sig_id": train.index, "fold": folds}).to_csv(
            output / "folds.csv", index=False
        )
        (output / "inference.py").write_bytes(model_source)
        per_label = losses(y, result.oof).mean(axis=0)
        label_counts = y.sum(axis=0)
        pd.DataFrame(
            {"label": labels.columns, "positive_rows": label_counts, "loss": per_label}
        ).to_csv(output / "label_losses.csv", index=False)
        pd.DataFrame(
            [y[validation].sum(axis=0) for _, validation in splits], columns=labels.columns
        ).to_csv(output / "fold_label_counts.csv", index_label="fold")

    with store.reserve(run_id) as reservation:
        model_source = MODEL.read_bytes()
        start = perf_counter()
        result = cross_fit_splits(len(y), splits, fit)
        validate_multilabel_probabilities(result.oof, *y.shape)
        if MODEL.read_bytes() != model_source or evaluation(folds, row) != runspec:
            raise ValueError("実行中に入力または推論ソースが変更されました")
        publish_cv_run(
            store,
            reservation,
            runspec,
            Trial(hypothesis, change, model_name, "kaggle-official", {"trial": trial}, inputs),
            y,
            splits,
            result,
            perf_counter() - start,
            write,
            save_query=False,
        )


def select(
    store: RunStore, groups: np.ndarray, y: np.ndarray, run_ids: list[str], output: Path
) -> None:
    selection = choose_run(store, groups, y, run_ids)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2)
        stream.write("\n")
    print(f"選定: {selection['run_id']}")


def choose_run(
    store: RunStore, groups: np.ndarray, y: np.ndarray, run_ids: list[str]
) -> dict[str, str | list[str]]:
    current = evaluation(make_folds(groups))
    records = [store.record(run_id) for run_id in run_ids]
    if len(records) != len(HYPOTHESES) or {record.params.get("trial") for record in records} != set(
        HYPOTHESES
    ):
        raise ValueError("候補には全ての比較実験を指定してください")
    scores = []
    for record in records:
        if (
            record.split_version != current.split_version
            or record.split_hash != current.split_hash
            or record.data_hash != current.data_hash
            or record.competition_id != current.competition_id
            or record.metric != current.metric
            or record.greater_is_better is not current.greater_is_better
            or record.seed != current.seed
            or record.params.get("trial") != record.run_id.partition("-")[0]
        ):
            raise ValueError(f"実験条件が一致しません: {record.run_id}")
        oof = store.load_array(record.run_id, "oof.npy")
        validate_multilabel_probabilities(oof, *y.shape)
        score = log_loss(y, oof)
        if not isclose(score, record.oof_score, rel_tol=0, abs_tol=1e-14):
            raise ValueError(f"OOF の再採点が実験ログと一致しません: {record.run_id}")
        scores.append((score, record.run_id))
    _, best = min(scores)
    source = verified_source(store, best)
    return {"run_ids": run_ids, "run_id": best, "source_sha256": sha256(source).hexdigest()}


def selected_run(selection_path: Path, groups: np.ndarray, y: np.ndarray) -> str:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != choose_run(RunStore(ROOT), groups, y, selection["run_ids"]):
        raise ValueError("選定記録が実験ログと一致しません")
    return str(selection["run_id"])


def verified_source(store: RunStore, run_id: str) -> bytes:
    expected = store.record(run_id).artifacts["inference.py"]
    source = store.artifact_path(run_id, "inference.py").read_bytes()
    if sha256(source).hexdigest() != expected:
        raise ValueError("推論ソースのハッシュが実験記録と一致しません")
    return source


def verify_run(run_id: str) -> tuple[dict[str, object], bytes]:
    store = RunStore(ROOT)
    record = store.record(run_id)
    if record.params.get("trial") != run_id.partition("-")[0]:
        raise ValueError("実験番号が実験記録と一致しません")
    if record.split_version != VERSION or record.data_hash != files_sha256(
        DATA / name for name in FILES
    ):
        raise ValueError("実験時とデータまたは分割が異なります")
    source = verified_source(store, run_id)
    return {
        "run_id": run_id,
        "data_hash": record.data_hash,
        "split_hash": record.split_hash,
        "source_sha256": sha256(source).hexdigest(),
        "params": record.params,
    }, source


def fit_submission(train: pd.DataFrame, labels: pd.DataFrame, run_id: str, output: Path) -> None:
    contract, source = verify_run(run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = output.with_suffix(".npz")
    manifest = output.with_suffix(".fit.json")
    if output.suffix != ".csv":
        raise ValueError("提出ファイルには .csv を指定してください")
    inference = {}
    exec(compile(source, "inference.py", "exec"), inference)
    with manifest.open("x", encoding="utf-8") as stream:
        created = []
        try:
            if output.exists() or weights.exists():
                raise FileExistsError("提出 CSV または NPZ が既にあります")
            trial = run_id.partition("-")[0]
            state = inference["fit_model"](train, labels.to_numpy(), trial)
            state["labels"] = np.asarray(labels.columns, dtype=str)
            state["contract"] = np.array(json.dumps(contract, sort_keys=True))
            created.append(output)
            inference["write_submission"](DATA, state, output)
            created.append(weights)
            np.savez_compressed(weights, **state)
            json.dump(
                {
                    "contract": contract,
                    "weights_sha256": file_sha256(weights),
                    "csv_sha256": file_sha256(output),
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        except BaseException:
            stream.close()
            for path in created:
                path.unlink(missing_ok=True)
            manifest.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("cv")
    run.add_argument("trial", choices=("E00", *HYPOTHESES))
    run.add_argument("--run-id", required=True)
    run.add_argument("--row", action="store_true")
    run.add_argument("--baseline")
    choose = sub.add_parser("select")
    choose.add_argument("run_ids", nargs="+")
    choose.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("selection", type=Path)
    final.add_argument("output", type=Path)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("run_id")
    baseline.add_argument("output", type=Path)
    args = parser.parse_args()
    train, labels, groups = load_training(DATA)
    store = RunStore(ROOT)
    if args.command == "prepare":
        prepare(train, labels, groups)
    elif args.command == "cv":
        cv(store, train, labels, groups, args.trial, args.run_id, args.row, args.baseline)
    elif args.command == "select":
        select(store, groups, labels.to_numpy(), args.run_ids, args.output)
    else:
        if args.command == "final":
            run_id = selected_run(args.selection, groups, labels.to_numpy())
        else:
            run_id = args.run_id
            if store.record(run_id).params["trial"] != "E01":
                raise ValueError("ベースラインは E01 を指定してください")
        fit_submission(train, labels, run_id, args.output)


if __name__ == "__main__":
    main()
