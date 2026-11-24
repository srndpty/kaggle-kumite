"""固定したグループ分割で仮説を比較する。"""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from model import SEED, f1_threshold, group_keys, predict, splits
from sklearn.metrics import average_precision_score, f1_score

from kumite.crossfit import CrossFitResult, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
SLUG = "quora-insincere-questions-classification"
TRIALS = {
    "E00": ("基準性能を測る", "単語 unigram と固定閾値", {}),
    "E01": ("閾値調整で Recall と F1 が上がる", "内側保留で閾値調整", {}),
    "E02": ("文字特徴で綴り変形に強くなる", "文字 3〜5-gram を追加", {"chars": True}),
    "E03": ("クラス重みの上積みは小さい", "balanced 重み", {"balanced": True}),
    "E04": (
        "大文字小文字を区別すると F1 が上がる",
        "単語の大小文字を保持",
        {"preserve_case": True},
    ),
}
SCORE = f1_score


def write_probabilities(out: Path, payload: dict[str, Any]) -> None:
    np.savez(out / "probabilities.npz", **payload)


def read_inputs(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data / "train.csv", dtype={"qid": str}, keep_default_na=False)
    test = pd.read_csv(data / "test.csv", dtype={"qid": str}, keep_default_na=False)
    if not all(frame["qid"].is_unique for frame in (train, test)):
        raise ValueError("qid が重複しています")
    if set(train["target"]) != {0, 1}:
        raise ValueError("target は 0 と 1 が必要です")
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="*")
    parser.add_argument("--run-id", help="1 試行を再実行する際の新しい run ID")
    parser.add_argument("--baseline-run-id", default="E00", help="E01 が参照する E00 の run ID")
    args = parser.parse_args()
    if set(args.trials) - TRIALS.keys():
        parser.error("未知の試行 ID")
    if args.run_id is not None and len(args.trials) != 1:
        parser.error("--run-id を指定するときは試行を 1 件指定してください")
    data = ROOT / "data"
    train, test = read_inputs(data)
    y = train["target"].to_numpy()
    text = train["question_text"].to_numpy()
    groups = group_keys(train["question_text"])
    duplicates = pd.DataFrame({"group": groups, "target": y}).groupby("group")["target"]
    print(
        f"train={len(train):,} test={len(test):,} positives={y.sum():,} "
        f"length={np.quantile(train['question_text'].str.len(), [0, 0.25, 0.5, 0.75, 1])} "
        f"duplicate_groups={int((duplicates.size() > 1).sum()):,} "
        f"conflicting_groups={int((duplicates.nunique() > 1).sum()):,} "
        f"shared_test={int(pd.Series(group_keys(test['question_text'])).isin(groups).sum()):,}",
        flush=True,
    )
    if not args.trials:
        return
    outer = splits(y, groups)
    inner = []
    roles = split_roles(len(y), outer)
    for fold, (pool, valid) in enumerate(outer):
        fit, tune = splits(y[pool], groups[pool])[0]
        inner.append((pool[fit], pool[tune], valid))
        roles[fold, pool[tune]] = 3
        print(
            f"fold={fold} fit={len(fit):,} tune={len(tune):,} "
            f"valid={len(valid):,} positives={y[valid].sum():,}",
            flush=True,
        )
    evaluation = Evaluation(
        "046",
        "F1",
        True,
        SCORE,
        files_sha256([data / "train.csv", data / "test.csv"]),
        "v1",
        array_sha256(roles),
        SEED,
    )
    store = RunStore(ROOT)
    for trial_id in args.trials:
        run_id = args.run_id or trial_id
        hypothesis, change, options = TRIALS[trial_id]
        with store.reserve(run_id) as reserved:
            start = perf_counter()
            probs = np.empty(len(y))
            coverage = np.empty(len(y))
            labels = np.empty(len(y), dtype=np.int8)
            tune_probs = []
            infos = []
            if (
                trial_id == "E01"
                and store.record(args.baseline_run_id).changed_one_thing != TRIALS["E00"][1]
            ):
                raise ValueError("E01 には E00 の確率を指定してください")
            baseline = (
                store.artifact_path(args.baseline_run_id, "probabilities.npz")
                if trial_id == "E01"
                else None
            )
            for fold, (fit, tune, valid) in enumerate(inner):
                if baseline is not None:
                    with np.load(baseline) as saved:
                        if (
                            str(saved["split_hash"]) != evaluation.split_hash
                            or str(saved["data_hash"]) != evaluation.data_hash
                        ):
                            raise ValueError("E00 の分割またはデータが一致しません")
                        pt = saved[f"tune_{fold}"]
                        pv = saved["oof"][valid]
                        cv = saved["coverage"][valid]
                else:
                    pt, pv, cv = predict(text, y, fit, tune, text[valid], **options)
                for values, n in ((pt, len(tune)), (pv, len(valid))):
                    if values.shape != (n,) or not np.all((values >= 0) & (values <= 1)):
                        raise ValueError("確率の形状または値が不正です")
                threshold = 0.5 if trial_id == "E00" else f1_threshold(y[tune], pt)
                policy = pv >= threshold
                probs[valid], labels[valid], coverage[valid] = pv, policy, cv
                tune_probs.append(pt)
                tp = int(np.count_nonzero(policy & (y[valid] == 1)))
                info = {
                    "threshold": threshold,
                    "ap": float(average_precision_score(y[valid], pv)),
                    "precision": tp / max(1, int(policy.sum())),
                    "recall": tp / int(y[valid].sum()),
                    "positive_rate": float(policy.mean()),
                }
                infos.append(info)
                print(f"{run_id} fold={fold} F1={SCORE(y[valid], policy):.5f} {info}", flush=True)

            payload: dict[str, np.ndarray | str] = {
                "oof": probs,
                "coverage": coverage,
                "split_hash": evaluation.split_hash,
                "data_hash": evaluation.data_hash,
                **{f"tune_{i}": p for i, p in enumerate(tune_probs)},
            }

            publish_cv_run(
                store,
                reserved,
                evaluation,
                Trial(
                    hypothesis,
                    change,
                    "TF-IDF + LR",
                    "official-csv",
                    dict(options),
                    store.input_hashes(args.baseline_run_id, "probabilities.npz")
                    if baseline
                    else {},
                ),
                y,
                outer,
                CrossFitResult(labels, np.empty(0), infos),
                perf_counter() - start,
                partial(write_probabilities, payload=payload),
                save_query=False,
            )


if __name__ == "__main__":
    main()
