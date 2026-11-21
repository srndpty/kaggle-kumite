"""OOF だけで提出候補を選び、被覆率別に比較する。"""

import argparse

from dataset import load_data
from segmentation import SEED, coverage_bins, image_scores
from train import ROOT

from kumite.bootstrap import group_bootstrap_diff
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="比較対象の run ID。先頭にベースラインを指定")
    args = parser.parse_args()
    store = RunStore(ROOT)
    records = [store.record(run_id) for run_id in args.runs]
    baseline = records[0]
    if any(
        (record.data_hash, record.split_hash) != (baseline.data_hash, baseline.split_hash)
        for record in records
    ):
        raise ValueError("データまたは分割が一致しません")
    candidate = max(records, key=lambda record: record.oof_score)
    data = load_data(ROOT / "data")
    base_scores = image_scores(
        data.masks, store.load_array(baseline.run_id, "oof.npy").astype(bool)
    )
    groups = store.load_array(baseline.run_id, "groups.npy")
    bins = coverage_bins(data.masks)
    selected = baseline
    print("run / OOF / 基準との差 [95% 区間] / 空 / 小面積 / その他")
    for record in records:
        scores = image_scores(data.masks, store.load_array(record.run_id, "oof.npy").astype(bool))
        diff = group_bootstrap_diff(
            base_scores,
            scores,
            groups,
            transform=lambda values: values,
            n_bootstrap=10_000,
            seed=SEED,
        )
        by_coverage = " / ".join(
            f"{scores[subset].mean():.6f}" for subset in (bins == 0, bins == 1, bins >= 2)
        )
        print(
            f"{record.run_id} / {scores.mean():.6f} / "
            f"{diff.point:+.6f} [{diff.low:+.6f}, {diff.high:+.6f}] / {by_coverage}"
        )
        if record.run_id == candidate.run_id and diff.low > 0:
            selected = candidate
    print(f"採用 {selected.run_id} / {store.artifact_path(selected.run_id, 'submission.csv')}")


if __name__ == "__main__":
    main()
