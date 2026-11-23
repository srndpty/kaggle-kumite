"""時系列 CV と、選定済みモデルの固定 rounds での再学習。"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import CARD, ID, TARGET, TIME, check_ids, data_files, features, load, make_splits
from select_run import selection
from sklearn.metrics import roc_auc_score

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.gbdt import lightgbm_binary
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, files_sha256, peak_memory_mb
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "base": "取引列のみ",
    "identity": "本人確認情報の追加",
    "history": "過去24時間・7日間のカード別取引件数",
    "clock": "24時間周期の時刻特徴",
    "relative_amount": "過去のカード別平均取引額に対する比率",
}


def auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred))


def check_predictions(pred: np.ndarray, size: int) -> None:
    if (
        pred.shape != (size,)
        or not np.isfinite(pred).all()
        or not ((pred >= 0) & (pred <= 1)).all()
    ):
        raise ValueError("予測の形状または確率が不正です")


def submission(sample: pd.DataFrame, ids: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    check_ids(sample)
    check_predictions(pred, len(ids))
    if list(sample) != [ID, TARGET] or set(sample[ID]) != set(ids) or not ids.is_unique:
        raise ValueError("提出データの ID または列がサンプルと一致しません")
    return sample.assign(isFraud=sample[ID].map(pd.Series(pred, index=ids)))


def card_status(ref: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    known = pd.MultiIndex.from_frame(rows[CARD]).isin(
        pd.MultiIndex.from_frame(ref[CARD].dropna().drop_duplicates())
    )
    return np.where(rows[CARD].isna().any(axis=1), "missing", np.where(known, "known", "new"))


def run(
    root: Path,
    run_id: str,
    trial: str,
    seed: int,
    source: str | None = None,
    selection_bytes: bytes | None = None,
) -> None:
    store = RunStore(root)
    data = root / "data"
    with store.reserve(run_id) as reservation:
        start = time.perf_counter()
        train = load(data, "train", trial == "identity")
        test = (
            load(data, "test", trial == "identity")
            if source
            else pd.read_csv(data / "test_transaction.csv", usecols=[TIME])
        )
        data_hash = files_sha256(data_files(data))
        splits, version = make_splits(train, test)
        roles = split_roles(len(train), splits)
        split_hash = array_sha256(roles)
        y = train[TARGET].to_numpy()
        params = {
            "num_boost_round": 500,
            "early_stopping": None,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": seed,
            "device_type": "cpu",
            "num_threads": 6,
            "deterministic": True,
            "force_col_wise": True,
        }

        def fit(tr: np.ndarray, va: np.ndarray) -> FoldOutput:
            x, valid, categories = features(train.iloc[tr], train.iloc[va], trial)
            output = lightgbm_binary(x, y[tr], valid, y[va], valid.iloc[:0], params, categories)
            check_predictions(output.valid, len(va))
            return output

        if source:
            record = store.record(source)
            if (record.data_hash, record.split_hash, record.seed, record.params) != (
                data_hash,
                split_hash,
                seed,
                {**params, "trial": trial},
            ):
                raise ValueError("元 run の入力・分割・設定が一致しません")
            x, query, categories = features(train, test, trial)
            pred = lightgbm_binary(x, y, query.iloc[:0], y[:0], query, params, categories).query
            sample = submission(pd.read_csv(data / "sample_submission.csv"), test[ID], pred)

            def write_submission(out: Path) -> None:
                sample.to_csv(out / "submission.csv", index=False)
                if selection_bytes is not None:
                    (out / "selection.json").write_bytes(selection_bytes)

            print(
                store.publish(
                    reservation,
                    replace(
                        record,
                        run_id=run_id,
                        hypothesis="固定設定で全行を再学習",
                        changed_one_thing="全学習行で再学習",
                        late_score=None,
                        artifacts={},
                        code_version=code_version(REPO_ROOT),
                        runtime_sec=time.perf_counter() - start,
                        peak_memory_mb=peak_memory_mb(),
                        inputs=store.input_hashes(source, "oof.npy"),
                    ),
                    write_submission,
                )
            )
            return

        result = cross_fit_splits(len(train), splits, fit)
        identity_ids = pd.read_csv(data / "train_identity.csv", usecols=[ID])[ID]
        diagnostics = []
        for fold, (tr, va) in enumerate(splits):
            rows = train.iloc[va]
            for name, groups in (
                ("identity", rows[ID].isin(identity_ids).to_numpy()),
                ("card", card_status(train.iloc[tr], rows)),
            ):
                for group in np.unique(groups):
                    mask = groups == group
                    labels = y[va][mask]
                    diagnostics.append(
                        {
                            "fold": fold,
                            "kind": name,
                            "group": str(group),
                            "n": len(labels),
                            "positive_rate": float(labels.mean()),
                            "auc": auc(labels, result.oof[va][mask])
                            if len(np.unique(labels)) == 2
                            else None,
                        }
                    )

        def write(out: Path) -> None:
            np.save(out / "split_roles.npy", roles)
            pd.DataFrame({ID: train[ID], TIME: train[TIME]}).to_csv(out / "rows.csv", index=False)
            (out / "diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
            (out / "windows.json").write_text(
                json.dumps(
                    [
                        {
                            "train_last": int(train[TIME].iloc[tr].max()),
                            "valid_first": int(train[TIME].iloc[va].min()),
                            "valid_last": int(train[TIME].iloc[va].max()),
                        }
                        for tr, va in splits
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )

        publish_cv_run(
            store,
            reservation,
            Evaluation("7", "ROC AUC", True, auc, data_hash, version, split_hash, seed),
            Trial(
                TRIALS[trial],
                TRIALS[trial],
                "LightGBM",
                "transactions-v1",
                {**params, "trial": trial},
            ),
            y,
            splits,
            result,
            time.perf_counter() - start,
            write,
            save_query=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("trial", choices=TRIALS, nargs="?")
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", help="ベースライン提出の元にする CV run")
    mode.add_argument("--selection", type=Path, help="最終提出を固定する選定記録")
    args = parser.parse_args()
    if args.selection:
        if args.trial is not None:
            parser.error("--selection と trial を同時に指定できません")
        contents = args.selection.read_bytes()
        payload = json.loads(contents)
        store = RunStore(ROOT)
        if selection(store, list(payload["candidates"])) != payload:
            raise ValueError("選定記録が CV 成果物と一致しません")
        record = store.record(payload["selected"])
        run(ROOT, args.run_id, record.params["trial"], record.seed, record.run_id, contents)
    elif args.trial is not None:
        run(ROOT, args.run_id, args.trial, args.seed, args.source)
    else:
        parser.error("trial または --selection を指定してください")


if __name__ == "__main__":
    main()
