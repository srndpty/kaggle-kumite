"""保存予測から fold 内と全体の AUC およびスコア分布を比較する。"""

import numpy as np
from dataset import read_ids
from run import DATA, OUTPUTS, ROOT, score
from scipy.stats import rankdata

from kumite.quality_gates import dense_oof, verify_cv_run
from kumite.runs import RunStore

RUNS = ("B00", "E01", "E02", "E03", "E04", "E05")


def main() -> None:
    ids, y = read_ids(DATA / "training_labels.csv", training=True)
    with np.load(OUTPUTS / "data-manifest.npz", allow_pickle=False) as manifest:
        if not np.array_equal(ids, manifest["train_ids"]):
            raise ValueError("マニフェストの学習 ID が CSV と一致しません")
        splits = [
            (np.flatnonzero(role == 1), np.flatnonzero(role == 2)) for role in manifest["roles"]
        ]
        data_hash = str(manifest["data_hash"].item())
    assess = dense_oof(y, splits, score, expected_shape=y.shape, data_hash=data_hash)
    store = RunStore(ROOT)
    summary = []
    print("| run | fold | AUC | 中央値 | 標準偏差 | 陰性 P10 / P50 / P90 | 陽性 P10 / P50 / P90 |")
    print("|---|---:|---:|---:|---:|---|---|")
    for run in RUNS:
        record = verify_cv_run(store, run, assess)
        oof = store.load_array(run, "oof.npy")
        ranks = np.empty_like(oof)
        for fold, (_, valid) in enumerate(splits):
            values = oof[valid]
            ranks[valid] = rankdata(values) / len(valid)
            quantiles = [
                " / ".join(
                    f"{point:.5f}"
                    for point in np.quantile(values[y[valid] == label], [0.1, 0.5, 0.9])
                )
                for label in (0, 1)
            ]
            print(
                f"| {run} | {fold + 1} | {record.fold_scores[fold]:.5f} | "
                f"{np.median(values):.5f} | {np.std(values):.5f} | {' | '.join(quantiles)} |"
            )
        summary.append(
            (
                run,
                record.oof_score,
                float(np.mean(record.fold_scores)),
                float(np.var(record.fold_scores)),
                score(y, ranks),
            )
        )

    print("\n| run | 全体 AUC | fold 平均 AUC | fold AUC の分散 | fold 内順位の全体 AUC |")
    print("|---|---:|---:|---:|---:|")
    for run, pooled, mean, variance, ranked in summary:
        print(f"| {run} | {pooled:.5f} | {mean:.5f} | {variance:.3e} | {ranked:.5f} |")


if __name__ == "__main__":
    main()
