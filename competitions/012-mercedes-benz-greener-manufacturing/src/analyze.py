"""保存済み run のペア比較と採用モデルの誤差分析。"""

import argparse

import numpy as np
import pandas as pd
from train import DATA, ROOT, SEED, categorical_columns, load_data, make_splits

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import validate_regression
from kumite.crossfit import split_roles
from kumite.provenance import array_sha256, file_sha256, files_sha256
from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="比較する run ID。先頭は基準")
    run_ids: list[str] = parser.parse_args().run_ids or ["E00", "E01", "E02", "E03", "E04"]
    base_id = run_ids[0]
    x, test, y, sample = load_data()
    groups, splits = make_splits(x)
    roles = split_roles(len(x), splits)
    _, repeated_splits = make_splits(x)
    assert np.array_equal(roles, split_roles(len(x), repeated_splits))
    assert np.all((roles == 2).sum(axis=0) == 1)
    assert all(not np.intersect1d(groups[a], groups[b]).size for a, b in splits)

    store = RunStore(ROOT)
    split_hash = array_sha256(roles)
    data_hash = files_sha256(
        DATA / name for name in ("train.csv", "test.csv", "sample_submission.csv")
    )
    preds = {}
    for run_id in run_ids:
        record = store.record(run_id)
        assert (record.data_hash, record.split_hash) == (data_hash, split_hash)
        run_dir = store.runs_dir / run_id
        for name, expected in record.artifacts.items():
            assert file_sha256(run_dir / name) == expected, f"{run_id}/{name}"
        pred = np.load(run_dir / "oof.npy")
        validate_regression(pred, len(y))
        query = np.load(run_dir / "test.npy")
        validate_regression(query, len(test))
        submission = pd.read_csv(run_dir / "submission.csv")
        assert list(submission.columns) == ["ID", "y"]
        assert submission["ID"].equals(sample["ID"])
        np.testing.assert_allclose(submission["y"].to_numpy(), query)
        preds[run_id] = pred

    base_loss = (preds[base_id] - y) ** 2
    accepted = [base_id]
    for run_id, pred in preds.items():
        if run_id == base_id:
            continue
        diff = group_bootstrap_diff(
            base_loss, (pred - y) ** 2, groups, transform=np.asarray, seed=SEED
        )
        print(f"{run_id} ΔMSE {diff.point:.5f} [{diff.low:.5f}, {diff.high:.5f}]")
        if diff.high < 0:
            accepted.append(run_id)
    winner = max(accepted, key=lambda run_id: store.record(run_id).oof_score)
    print(f"採用: {winner}")

    rare = np.zeros(len(y), dtype=bool)
    unseen = np.zeros(len(y), dtype=bool)
    cat = categorical_columns(x)
    for train_idx, valid_idx in splits:
        for name in cat:
            counts = x.iloc[train_idx][name].value_counts()
            frequency = x.iloc[valid_idx][name].map(counts).fillna(0).to_numpy()
            rare[valid_idx] |= (frequency > 0) & (frequency < 5)
            unseen[valid_idx] |= frequency == 0
    tail = y >= np.quantile(y, 0.95)
    ids = pd.read_csv(DATA / "train.csv", usecols=["ID"])["ID"]
    error = preds[winner] - y

    def summarize(name: str, mask: np.ndarray) -> None:
        rows = error[mask]
        if len(rows):
            print(
                f"{name}: n={len(rows)} MAE={np.abs(rows).mean():.4f} "
                f"bias={rows.mean():.4f} SSE割合={(rows**2).sum() / (error**2).sum():.4f}"
            )

    summarize("目的変数の上位5%", tail)
    summarize("希少な既知水準あり・未知水準なし", rare & ~unseen)
    summarize("未知水準あり", unseen)
    summarize("全て5件以上", ~rare & ~unseen)
    for band in pd.qcut(ids, 4).cat.categories:
        mask = ((ids > band.left) & (ids <= band.right)).to_numpy()
        summarize(f"ID {band}", mask)
        print(f"  y平均={y[mask].mean():.4f} X5水準数={x.loc[mask, 'X5'].nunique()}")


if __name__ == "__main__":
    main()
