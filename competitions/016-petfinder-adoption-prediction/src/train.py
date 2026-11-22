"""公式データで EDA と比較実験を実行する。"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from dataset import (
    GROUP,
    ID,
    TARGET,
    connected_groups,
    image_features,
    image_paths,
    read_data,
    tabular,
)
from model import (
    DESCRIPTION_COMPONENTS,
    FIXED,
    PARAMS,
    SEED,
    calibrate,
    classify,
    evaluate,
    learn_thresholds,
    make_folds,
    qwk,
)
from sklearn.metrics import confusion_matrix

from kumite.crossfit import CrossFitResult, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import find_record
from kumite.provenance import array_sha256, file_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "dummy": ("定数予測で採点経路を確認する", "クラス2の定数予測"),
    "baseline": ("表特徴による基準", "表特徴と固定閾値"),
    "regression": ("順序を学習すると大きな誤分類が減る", "学習目的を回帰に変更"),
    "threshold": ("閾値調整で QWK が改善する", "内側 OOF で閾値を調整"),
    "image": ("画像統計に PhotoAmt を超える情報がある", "画像統計を追加"),
    "description": ("掲載本文に表特徴を超える情報がある", "掲載本文の TF-IDF と SVD を追加"),
}


def submission(sample: pd.DataFrame, ids: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    if pred.shape != (len(ids),) or not np.isin(pred, range(5)).all():
        raise ValueError("提出値は 0〜4 の整数クラスです")
    result = pd.Series(pred.astype(int), index=ids).reindex(sample[ID])
    if result.isna().any() or not result.index.equals(pd.Index(sample[ID])):
        raise ValueError("提出行がサンプルと一致しません")
    return result.rename(TARGET).reset_index()


def baseline_scores(
    store: RunStore,
    assignments: pd.DataFrame,
    data_hash: str,
    split_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    baseline = find_record(store.records(), "E01")
    if (baseline.data_version, baseline.params, baseline.data_hash, baseline.split_hash) != (
        "baseline",
        PARAMS,
        data_hash,
        split_hash,
    ):
        raise ValueError("E01 の学習条件が閾値実験と一致しません")
    files = [store.runs_dir / "E01" / name for name in ("oof.csv", "test_scores.npy")]
    inputs = {}
    for path in files:
        digest = file_sha256(path)
        if digest != baseline.artifacts[path.name]:
            raise ValueError(f"E01 の成果物のハッシュが一致しません: {path.name}")
        inputs[f"E01/{path.name}"] = digest
    oof = pd.read_csv(files[0], dtype={ID: str}, converters={"score": float})
    if not oof[[ID, TARGET, "fold"]].equals(assignments[[ID, TARGET, "fold"]]):
        raise ValueError("E01 の OOF と PetID、ラベル、fold が一致しません")
    return oof["score"].to_numpy(), np.load(files[1]), inputs


def prepare(root: Path, variant: str):
    train, test, sample, paths = read_data(root)
    include_images = variant == "image"
    train_images, train_hashes, unreadable = image_features(
        image_paths(root, "train"), train[ID], summarize=include_images or variant == "eda"
    )
    test_images, test_hashes, _ = image_features(
        image_paths(root, "test"), test[ID], summarize=include_images
    )
    groups = connected_groups(train, train_hashes)
    y = train[TARGET].to_numpy()
    folds = make_folds(y, groups, 5)
    data_hash = sha256(
        json.dumps([files_sha256(paths), train_hashes, test_hashes], sort_keys=True).encode()
    ).hexdigest()
    x, query = tabular(train), tabular(test)
    if include_images:
        x, query = x.join(train_images), query.join(test_images)
    if variant == "description":
        x["Description"] = train["Description"].fillna("")
        query["Description"] = test["Description"].fillna("")
    assignments = train[[ID, GROUP, TARGET, "PhotoAmt"]].assign(group=groups, fold=folds)
    if variant == "eda":
        outputs = ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        train_images.assign(PetID=train[ID]).to_csv(outputs / "eda_images.csv", index=False)
        file_counts = pd.Series([name.rsplit("-", 1)[0] for name in train_hashes]).value_counts()
        print(
            json.dumps(
                {
                    "class_counts": train[TARGET].value_counts().sort_index().to_dict(),
                    "rescuers": int(train[GROUP].nunique()),
                    "connected_groups": len(np.unique(groups)),
                    "max_listings_per_rescuer": int(train[GROUP].value_counts().max()),
                    "duplicate_image_files": len(train_hashes) - len(set(train_hashes.values())),
                    "listings_without_readable_images": int(
                        (train_images["readable_images"] == 0).sum()
                    ),
                    "photo_count_mismatch": int(
                        (train["PhotoAmt"] != train[ID].map(file_counts).fillna(0)).sum()
                    ),
                    "unreadable_images": unreadable,
                },
                indent=2,
            )
        )
        print(pd.crosstab(folds, y))
    return assignments, test[ID], sample, x, query, data_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["eda", *TRIALS])
    parser.add_argument("--run-id")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    variant = args.variant
    if variant == "eda":
        prepare(args.data, variant)
        return
    if not args.run_id:
        parser.error("学習には --run-id が必要です")
    store = RunStore(ROOT)
    with store.reserve(args.run_id) as reserved:
        started = perf_counter()
        assignments, ids, sample, x, query, data_hash = prepare(args.data, variant)
        y = assignments[TARGET].to_numpy()
        groups = assignments["group"].to_numpy()
        folds = assignments["fold"].to_numpy()
        split_hash = array_sha256(folds)
        inputs: dict[str, str] = {}
        if variant == "dummy":
            result = CrossFitResult(
                np.full((len(y), 2), 2), np.full(len(query), 2), [{} for _ in range(5)]
            )
        elif variant == "threshold":
            oof, scores, inputs = baseline_scores(store, assignments, data_hash, split_hash)
            result = calibrate(x, y, groups, folds, oof, scores)
        else:
            result = evaluate(x, y, folds, query, variant, workers=5)
        thresholds = learn_thresholds(y, result.oof[:, 0]) if variant == "threshold" else FIXED
        oof_classes = result.oof[:, 1].astype(int)
        test_classes = classify(result.query, thresholds)
        csv = submission(sample, ids, test_classes)

        def write(out: Path) -> None:
            assignments.assign(score=result.oof[:, 0], prediction=oof_classes).to_csv(
                out / "oof.csv", index=False
            )
            np.save(out / "test_scores.npy", result.query)
            csv.to_csv(out / "submission.csv", index=False)
            (out / "diagnostics.json").write_text(
                json.dumps(
                    {
                        "submission_thresholds": thresholds.tolist(),
                        "confusion_matrix": confusion_matrix(
                            y, oof_classes, labels=range(5)
                        ).tolist(),
                    }
                ),
                encoding="utf-8",
            )

        hypothesis, change = TRIALS[variant]
        evaluation = Evaluation(
            competition_id="016",
            metric="QWK",
            greater_is_better=True,
            score=qwk,
            data_hash=data_hash,
            split_version="rescuer-image-v1",
            split_hash=split_hash,
            seed=SEED,
        )
        params = {"class": 2} if variant == "dummy" else dict(PARAMS)
        if variant == "description":
            params.update(description_svd_components=DESCRIPTION_COMPONENTS, word_bigrams=True)
        trial = Trial(
            hypothesis,
            change,
            "定数" if variant == "dummy" else "CatBoost",
            variant,
            params,
            inputs,
        )
        publish_cv_run(
            store,
            reserved,
            evaluation,
            trial,
            y,
            folds_to_splits(folds),
            CrossFitResult(oof_classes, test_classes, result.infos),
            perf_counter() - started,
            write,
        )


if __name__ == "__main__":
    main()
