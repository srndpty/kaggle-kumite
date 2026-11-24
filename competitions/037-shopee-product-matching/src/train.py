"""タイトル検索の外側 CV を実行し、候補と OOF を記録する。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import Booster
from matching import (
    Candidates,
    accepted,
    fit_reranker,
    fit_text,
    mutual_edges,
    rerank,
    row_metrics,
    top_candidates,
    tune_threshold,
)
from split import SEED, SPLIT_VERSION, assign_folds

from kumite.experiments import ExperimentRecord
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "E00": ("文字 TF-IDF で同じ商品を検索できる", "文字 3–5 gram TF-IDF"),
    "E01": (
        "単語類似度で型番や商品名が一致する候補が増える",
        "単語 1–2 gram cosine を等重みで追加",
    ),
    "E02": ("画像検索でタイトル表記が違う正解を拾える", "画像 cosine 上位50件を候補に追加"),
    "E03": ("同じ候補から誤一致を減らせる", "E02 のペアを LightGBM で再スコアリング"),
    "E04": ("片方向の採用には誤一致が多い", "E03 の採用を双方向のペアに限定"),
}


def retrieve(
    fit: pd.DataFrame, gallery: pd.DataFrame, trial: str, images: np.ndarray | None
) -> Candidates:
    return top_candidates(
        fit_text(fit["title"].tolist(), words=trial == "E01"),
        gallery["title"].tolist(),
        images=None if images is None else images[gallery.index.to_numpy()],
        hashes=gallery["image_phash"].to_numpy(),
    )


def inner_candidates(
    train: pd.DataFrame, trial: str, images: np.ndarray | None
) -> list[tuple[Candidates, np.ndarray]]:
    inner = assign_folds(train["component"].to_numpy(), 3)
    folds: list[tuple[Candidates, np.ndarray]] = []
    for fold in range(3):
        tune = train.loc[inner == fold]
        folds.append(
            (
                retrieve(train.loc[inner != fold], tune, trial, images),
                tune["label_group"].to_numpy(),
            )
        )
    return folds


def calibrate(
    train: pd.DataFrame, trial: str, images: np.ndarray | None
) -> tuple[float, Booster | None]:
    calibration = inner_candidates(train, trial, images)
    model = None
    if trial == "E03":
        model = fit_reranker(calibration)
        calibration = [
            (rerank(candidate, fit_reranker(calibration[:i] + calibration[i + 1 :])), labels)
            for i, (candidate, labels) in enumerate(calibration)
        ]
    return tune_threshold(calibration), model


def load_rows(train_path: Path, split_path: Path) -> pd.DataFrame:
    split = pd.read_csv(split_path)
    rows = pd.read_csv(train_path).merge(split, on="posting_id", how="left", validate="one_to_one")
    if (
        len(rows) != len(split)
        or set(rows["fold"]) != set(range(5))
        or rows["component"].isna().any()
    ):
        raise ValueError("学習行と5 fold の割当が一致しません")
    for group in ("label_group", "image", "component"):
        if rows.groupby(group)["fold"].nunique().max() != 1:
            raise ValueError(f"{group} が複数 fold にまたがっています")
    rows["title"] = rows["title"].fillna("")
    return rows.sort_values("posting_id").reset_index(drop=True)


def load_images(
    rows: pd.DataFrame, train_path: Path, cache_path: Path
) -> tuple[np.ndarray, dict[str, str]]:
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["csv_sha256"] != file_sha256(train_path):
        raise ValueError("画像埋め込みの元 CSV が一致しません")
    if manifest["embeddings_sha256"] != file_sha256(cache_path):
        raise ValueError("画像埋め込みのハッシュが一致しません")
    with np.load(cache_path) as cache:
        lookup = {name: i for i, name in enumerate(cache["names"].tolist())}
        images = cache["values"][[lookup[name] for name in rows["image"]]]
    return images, {name: value for name, value in manifest.items() if name.endswith("_sha256")}


def predict_fold(
    train: pd.DataFrame, valid: pd.DataFrame, trial: str, images: np.ndarray | None
) -> tuple[Candidates, float]:
    threshold, model = calibrate(train, trial, images)
    candidates = retrieve(train, valid, trial, images)
    if model is not None:
        candidates = rerank(candidates, model)
    return candidates, threshold


def evaluate(
    rows: pd.DataFrame, trial: str, images: np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    measurements: list[pd.DataFrame] = []
    pairs: list[pd.DataFrame] = []
    for fold in range(5):
        train = rows.loc[rows["fold"] != fold]
        valid = rows.loc[rows["fold"] == fold]
        candidates, threshold = predict_fold(train, valid, trial, images)
        ids = valid["posting_id"].to_numpy()
        measurements.append(
            row_metrics(candidates, valid["label_group"].to_numpy(), threshold).assign(
                posting_id=ids, fold=fold, threshold=threshold
            )
        )
        finite = np.isfinite(candidates.scores.ravel())
        pairs.append(
            pd.DataFrame(
                {
                    "posting_id": np.repeat(ids, candidates.indices.shape[1])[finite],
                    "candidate_id": ids[candidates.indices].ravel()[finite],
                    "score": candidates.scores.ravel()[finite],
                    "accepted": accepted(candidates, threshold).ravel()[finite],
                }
            )
        )
        print(f"fold {fold}: {measurements[-1]['f1'].mean():.6f}", flush=True)
    return pd.concat(measurements, ignore_index=True), pd.concat(pairs, ignore_index=True)


def evaluate_mutual(
    rows: pd.DataFrame, store: RunStore, base_run: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(store.artifact_path(base_run, "candidates.csv.gz"))
    oof = pd.read_csv(store.artifact_path(base_run, "oof.csv")).set_index("posting_id")
    chosen = pairs.loc[pairs["accepted"]]
    mutual = mutual_edges(chosen["posting_id"].to_numpy(), chosen["candidate_id"].to_numpy())
    pairs.loc[chosen.index, "accepted"] = mutual
    retained = chosen.loc[mutual]
    labels = rows.set_index("posting_id")["label_group"]
    correct = retained["posting_id"].map(labels) == retained["candidate_id"].map(labels)
    true_count = retained.loc[correct, "posting_id"].value_counts().reindex(oof.index, fill_value=0)
    pred_count = retained["posting_id"].value_counts().reindex(oof.index, fill_value=0)
    sizes = labels.map(labels.value_counts()).reindex(oof.index)
    oof["f1"] = 2 * (1 + true_count) / (1 + pred_count + sizes)
    return oof.reset_index(), pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--base-run", default="E03")
    args = parser.parse_args()
    run_id = args.run_id or args.trial
    store = RunStore(ROOT)
    train_path = ROOT / "data/train.csv"
    split_path = ROOT / "outputs" / f"{SPLIT_VERSION}.csv"
    with store.reserve(run_id) as reservation:
        start = perf_counter()
        version = code_version(ROOT.parent.parent)
        rows = load_rows(train_path, split_path)
        images = None
        inputs: dict[str, str] = {}
        data_hash, split_hash = file_sha256(train_path), file_sha256(split_path)
        params = {"trial": args.trial, "top_k": 50, "inner_folds": 3}
        if args.trial == "E04":
            base = store.record(args.base_run)
            if base.params["trial"] != "E03" or (base.data_hash, base.split_hash) != (
                data_hash,
                split_hash,
            ):
                raise ValueError("E04 は同じデータと分割の E03 が必要です")
            oof, pairs = evaluate_mutual(rows, store, args.base_run)
            inputs = base.inputs | {
                f"{args.base_run}/{name}": base.artifacts[name]
                for name in ("oof.csv", "candidates.csv.gz")
            }
            params["base_run"] = args.base_run
        else:
            if args.trial in {"E02", "E03"}:
                images, inputs = load_images(
                    rows, train_path, ROOT / "outputs/train-embeddings.npz"
                )
            oof, pairs = evaluate(rows, args.trial, images)
        metrics = oof[["f1", "candidate_recall", "oracle_f1"]].mean().to_dict()
        hypothesis, change = TRIALS[args.trial]
        record = ExperimentRecord(
            competition_id=ROOT.name,
            run_id=run_id,
            metric="mean_f1",
            greater_is_better=True,
            data_version="titles_v1",
            data_hash=data_hash,
            split_version=SPLIT_VERSION,
            split_hash=split_hash,
            seed=SEED,
            hypothesis=hypothesis,
            changed_one_thing=change,
            model="LightGBM" if args.trial in {"E03", "E04"} else "類似度検索",
            params=params,
            fold_scores=oof.groupby("fold")["f1"].mean().tolist(),
            oof_score=float(metrics["f1"]),
            runtime_sec=perf_counter() - start,
            code_version=version,
            inputs=inputs,
        )

        def write(output: Path) -> None:
            oof.to_csv(output / "oof.csv", index=False)
            pairs.to_csv(output / "candidates.csv.gz", index=False)
            rows[["posting_id", "component", "fold"]].to_csv(output / "split.csv", index=False)
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(store.publish(reservation, record, write))


if __name__ == "__main__":
    main()
