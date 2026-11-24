"""課題単位 CV、比較実験、選定と誤差分析。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from model import CHANGES, FILES, TARGETS, dense_features, load, predict, tokens
from sklearn.model_selection import KFold, LeaveOneGroupOut

from kumite import text_features
from kumite.crossfit import FoldOutput, Split, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
PRIMARY_SPLIT = "prompt-loo-v1"


def rmse_columns(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y - prediction) ** 2, axis=0))


def mcrmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(rmse_columns(y, prediction).mean())


def make_splits(train: pd.DataFrame, *, random: bool = False) -> list[Split]:
    normalized = pd.Series([" ".join(tokens(text)) for text in train["text"].tolist()])
    if random:
        splits = KFold(n_splits=4, shuffle=True, random_state=SEED).split(train)
    else:
        splits = LeaveOneGroupOut().split(train, groups=train["prompt_id"])
    return [
        (fit[~normalized.iloc[fit].isin(normalized.iloc[valid]).to_numpy()], valid)
        for fit, valid in splits
    ]


def audit(train: pd.DataFrame, test: pd.DataFrame) -> None:
    lengths = dense_features(train, "E00")[:, 0]
    summary = train.assign(
        words=lengths,
        exact_source=train.apply(lambda row: tokens(row.text) == tokens(row.prompt_text), axis=1),
    )
    by_prompt = summary.groupby("prompt_id")
    print(f"train={len(train)} test={len(test)} test_prompts={test['prompt_id'].nunique()}")
    print(by_prompt[["words", *TARGETS]].quantile(np.array([0.1, 0.5, 0.9])).to_string())
    print(by_prompt.agg(rows=("student_id", "size"), exact_source=("exact_source", "sum")))
    normalized = pd.Series([" ".join(tokens(text)) for text in train["text"].tolist()])
    print(f"完全一致グループに含まれる行数: {int(normalized.duplicated(keep=False).sum())}")
    for fit, valid in make_splits(train):
        print(
            f"{train.iloc[valid[0]]['prompt_id']}: 検証 {len(valid)}, 重複除外 {len(train) - len(fit) - len(valid)}"
        )


def evaluation(roles: np.ndarray, random: bool) -> Evaluation:
    return Evaluation(
        "050",
        "MCRMSE",
        False,
        mcrmse,
        files_sha256(DATA / name for name in FILES),
        "row-random-v1" if random else PRIMARY_SPLIT,
        sha256(roles.tobytes()).hexdigest(),
        SEED,
    )


def source_hash() -> str:
    return files_sha256([ROOT / "src" / "model.py", Path(text_features.__file__)])


def run_cv(train: pd.DataFrame, test: pd.DataFrame, trial: str, run_id: str) -> None:
    if run_id.partition("-")[0] != trial:
        raise ValueError("run ID と試行が一致しません")
    random = trial == "E04"
    splits = make_splits(train, random=random)
    roles = split_roles(len(train), splits)
    if not ((roles == 2).sum(axis=0) == 1).all():
        raise ValueError("すべての行を1回ずつ検証してください")
    config = evaluation(roles, random)
    y = train[TARGETS].to_numpy()
    store = RunStore(ROOT)
    start = perf_counter()

    def fit_fold(fit: np.ndarray, valid: np.ndarray) -> FoldOutput:
        query = pd.concat([train.iloc[valid], test], ignore_index=True)
        predictions = predict(train.iloc[fit], query, trial)
        mean_prediction = np.broadcast_to(y[fit].mean(axis=0), y[valid].shape)
        return FoldOutput(
            predictions[: len(valid)],
            predictions[len(valid) :],
            {
                "train_rows": len(fit),
                "valid_rows": len(valid),
                "mean_reference": mcrmse(y[valid], mean_prediction),
            },
        )

    with store.reserve(run_id) as reservation:
        result = cross_fit_splits(len(train), splits, fit_fold)
        mean_oof = np.empty_like(y)
        for fit, valid in splits:
            mean_oof[valid] = y[fit].mean(axis=0)
        print(f"列平均の参照 OOF MCRMSE: {mcrmse(y, mean_oof):.6f}")
        print(f"目的列別 RMSE: {rmse_columns(y, result.oof)}")

        def write_artifacts(output: Path) -> None:
            np.save(output / "split_roles.npy", roles)
            for source in (ROOT / "src" / "model.py", Path(text_features.__file__)):
                (output / source.name).write_bytes(source.read_bytes())
            np.save(output / "mean_oof.npy", mean_oof)
            train[["student_id", "prompt_id"]].to_csv(output / "oof_rows.csv", index=False)

        publish_cv_run(
            store,
            reservation,
            config,
            Trial(
                CHANGES[trial],
                CHANGES[trial],
                "Ridge",
                "kaggle-official",
                {"trial": trial, "alpha": 1.0, "solver": "lsqr", "source_hash": source_hash()},
            ),
            y,
            splits,
            result,
            perf_counter() - start,
            write_artifacts,
        )


def select(train: pd.DataFrame, run_ids: list[str]) -> dict[str, object]:
    roles = split_roles(len(train), make_splits(train))
    config = evaluation(roles, False)
    store = RunStore(ROOT)
    y = train[TARGETS].to_numpy()
    comparisons: dict[str, dict[str, object]] = {}
    order: list[tuple[float, int, str]] = []
    words = dense_features(train, "E00")[:, 0]
    short, long = np.quantile(words, [1 / 3, 2 / 3])
    groups = {
        **{
            f"prompt/{p}": (train["prompt_id"] == p).to_numpy()
            for p in sorted(train["prompt_id"].unique())
        },
        "length/short": words <= short,
        "length/middle": (words > short) & (words <= long),
        "length/long": words > long,
    }
    for run_id in run_ids:
        record = store.record(run_id)
        if (record.data_hash, record.split_version, record.split_hash) != (
            config.data_hash,
            config.split_version,
            config.split_hash,
        ):
            raise ValueError(f"{run_id}: 比較条件が一致しません")
        prediction = np.load(store.artifact_path(run_id, "oof.npy"), allow_pickle=False)
        score = mcrmse(y, prediction)
        if score != record.oof_score:
            raise ValueError(f"{run_id}: OOF スコアが一致しません")
        order.append((score, int(record.params["trial"] != "E00"), run_id))
        comparisons[run_id] = {
            "mcrmse": score,
            "rmse": dict(zip(TARGETS, rmse_columns(y, prediction).tolist(), strict=True)),
            "groups": {
                name: {"rows": int(mask.sum()), "mcrmse": mcrmse(y[mask], prediction[mask])}
                for name, mask in groups.items()
                if mask.any()
            },
        }
    return {
        "selected_run": min(order)[2],
        "data_hash": config.data_hash,
        "split_hash": config.split_hash,
        "length_cutoffs": [short, long],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    cv = commands.add_parser("cv")
    cv.add_argument("trial", choices=CHANGES)
    cv.add_argument("--run-id")
    selection = commands.add_parser("select")
    selection.add_argument("runs", nargs="+")
    selection.add_argument("--output", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    train, test = load(DATA)
    if args.command == "audit":
        audit(train, test)
    elif args.command == "cv":
        run_cv(train, test, args.trial, args.run_id or args.trial)
    else:
        selection_data = select(train, args.runs)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(selection_data, indent=2) + "\n")


if __name__ == "__main__":
    main()
