"""比較可能な OOF run から最終モデルを固定し、選定の証跡を保存する。"""

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dataset import FILES, SEED, labels_and_groups, load_train
from evidence import checked_oof
from metrics import IDENTITIES, aggregate
from train import ROOT

from kumite.provenance import files_sha256
from kumite.runs import RunStore

BOOTSTRAPS = 399


def tie_bounds(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.searchsorted(reference, query, side="left"),
        np.searchsorted(reference, query, side="right"),
    )


def weighted_before(prefix: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """同点を半分として、各スコアより下の再標本件数を返す。"""
    left, right = bounds
    return (prefix[left] + prefix[right]) * 0.5


def rank_auc(
    labels: np.ndarray,
    groups: np.ndarray,
    prediction: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """スコア順を一度作り、再標本ごとに重み付き AUC を計算する。"""
    order = np.argsort(prediction)
    positive, negative = order[labels[order, 0]], order[~labels[order, 0]]
    positive_scores, negative_scores = prediction[positive], prediction[negative]
    positive_groups, negative_groups = groups[positive], groups[negative]
    overall_bounds = tie_bounds(negative_scores, positive_scores)
    identities = []
    for index in range(1, len(IDENTITIES) + 1):
        pos = np.flatnonzero(labels[positive, index])
        neg = np.flatnonzero(labels[negative, index])
        subgroup_bounds = tie_bounds(negative_scores[neg], positive_scores[pos])
        bpsn_bounds = tie_bounds(positive_scores, negative_scores[neg])
        identities.append((pos, neg, subgroup_bounds, bpsn_bounds))

    def evaluate(weights: np.ndarray) -> np.ndarray:
        pos_weights, neg_weights = weights[positive_groups], weights[negative_groups]
        pos_prefix = np.concatenate(([0], np.cumsum(pos_weights)))
        neg_prefix = np.concatenate(([0], np.cumsum(neg_weights)))
        total_pos, total_neg = pos_prefix[-1], neg_prefix[-1]
        before_pos = weighted_before(neg_prefix, overall_bounds)
        numerators = [np.sum(pos_weights * before_pos)]
        denominators = [total_pos * total_neg]
        for pos, neg, subgroup_bounds, bpsn_bounds in identities:
            group_pos, group_neg = pos_weights[pos], neg_weights[neg]
            subgroup_neg_prefix = np.concatenate(([0], np.cumsum(group_neg)))
            group_pos_count, group_neg_count = group_pos.sum(), subgroup_neg_prefix[-1]
            subgroup = np.sum(group_pos * weighted_before(subgroup_neg_prefix, subgroup_bounds))
            # 全陽性×集団陰性、集団陽性×全陰性から、集団内のペアを引く。
            bpsn = np.sum(group_neg * (total_pos - weighted_before(pos_prefix, bpsn_bounds)))
            bnsp = np.sum(group_pos * before_pos[pos])
            numerators.extend((subgroup, bpsn - subgroup, bnsp - subgroup))
            denominators.extend(
                (
                    group_pos_count * group_neg_count,
                    (total_pos - group_pos_count) * group_neg_count,
                    group_pos_count * (total_neg - group_neg_count),
                )
            )
        denominators_array = np.asarray(denominators)
        if (denominators_array == 0).any():
            raise ValueError("再標本内の AUC に両クラスが必要です")
        return np.asarray(numerators) / denominators_array

    return evaluate


def bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    count: int = BOOTSTRAPS,
) -> tuple[float, float]:
    n_groups = int(groups.max()) + 1
    aucs = np.empty((count, 2, 1 + 3 * len(IDENTITIES)))
    ranked = [rank_auc(labels, groups, prediction) for prediction in (base, candidate)]
    rng = np.random.default_rng(SEED)
    for sample in range(count):
        # 全指標・両モデルで同じグループ集合を再標本化する。
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        for model, evaluate in enumerate(ranked):
            aucs[sample, model] = evaluate(weights)
        if (sample + 1) % 100 == 0:
            print(f"bootstrap: {sample + 1}/{count}", flush=True)
    scores = aggregate(aucs).mean(axis=-1)
    low, high = np.quantile(scores[:, 1] - scores[:, 0], [0.025, 0.975])
    return float(low), float(high)


def select(root: Path, base_id: str, candidate_ids: list[str]) -> dict:
    if not candidate_ids or len({base_id, *candidate_ids}) != len(candidate_ids) + 1:
        raise ValueError("基準と候補には異なる run ID が必要です")
    data = root / "data"
    data_hash = files_sha256(data / name for name in FILES)
    store = RunStore(root)
    records = {record.run_id: record for record in store.records()}
    base = records[base_id]
    if base.params.get("trial") != "E01" or base.data_hash != data_hash:
        raise ValueError("ベースラインは現在の公式データを使った E01 が必要です")
    train = load_train(data)
    labels, groups = labels_and_groups(train)
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for run_id in (base_id, *candidate_ids):
        record = records[run_id]
        predictions[run_id], scores[run_id] = checked_oof(
            store, record, data_hash, base.split_hash, train["id"], labels
        )
        hashes[run_id] = record.artifacts["oof.npy"]

    best = max(candidate_ids, key=scores.__getitem__)
    delta = scores[best] - scores[base_id]
    interval = None
    selected = base_id
    if delta > 0:
        interval = bootstrap(labels, groups, predictions[base_id], predictions[best])
        if interval[0] > 0:
            selected = best
    store.artifact_path(selected, "submission.csv")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "data_hash": data_hash,
        "split_hash": base.split_hash,
        "oof_hashes": hashes,
        "baseline": base_id,
        "best_candidate": best,
        "scores": scores,
        "score_delta": delta,
        "bootstrap_interval_95": interval,
        "bootstrap_samples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "selected": selected,
        "submission_hash": records[selected].artifacts["submission.csv"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    path = ROOT / "outputs" / "selection.json"
    if path.exists():
        raise FileExistsError(path)
    selection = select(ROOT, args.baseline, args.candidates)
    with path.open("x", encoding="utf-8") as out:
        json.dump(selection, out, indent=2)
    print(json.dumps(selection, indent=2))
