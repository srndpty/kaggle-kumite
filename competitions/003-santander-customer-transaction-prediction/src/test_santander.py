from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analyze import analyze
from features import log_counts, transform
from scipy.sparse import csr_matrix
from train import fold_predict, make_splits, model_params, run

from kumite.checks import check_label_independence, shift_labels
from kumite.runs import RunStore


def test_counts_exclude_self_and_handle_unseen_values() -> None:
    fit, query = log_counts(np.array([[1.0], [1.0], [2.0]]), np.array([[1.0], [0.0], [3.0]]))
    np.testing.assert_allclose(fit[:, 0], [np.log(2), np.log(2), 0])
    np.testing.assert_allclose(query[:, 0], [np.log(3), 0, 0])


def test_bin_boundaries_are_independent_of_query() -> None:
    train = np.arange(32, dtype=float).reshape(-1, 1)
    a, _ = transform("E03", train, np.array([[100.0]]))
    b, _ = transform("E03", train, np.array([[-100.0]]))
    np.testing.assert_array_equal(csr_matrix(a).toarray(), csr_matrix(b).toarray())


def test_duplicate_groups_are_kept_together() -> None:
    x = np.repeat(np.arange(50, dtype=float), 2).reshape(-1, 1)
    y = np.tile([0, 1], 50)
    splits = make_splits(x, y, seed=42)
    for train, valid in splits:
        assert not np.intersect1d(x[train], x[valid]).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate([valid for _, valid in splits])), np.arange(len(x))
    )


def test_validation_labels_do_not_change_fold_predictions() -> None:
    x = np.arange(120, dtype=float).reshape(-1, 2)
    y = np.tile([0, 1], 30)
    train, valid = make_splits(x, y, seed=42)[0]

    def build(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = fold_predict("E02", model_params("E02", 42), x[train], labels[train], x[valid], x[:2])
        return out.valid, out.query

    mask = np.zeros(len(y), dtype=bool)
    mask[valid] = True
    check_label_independence(build, y, shift_labels(y, mask, 2))


@pytest.mark.parametrize("variant", ["E01", "E02", "E03", "E04", "E05", "E06"])
def test_run_publishes_aligned_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    monkeypatch.setattr("train.ROOT", tmp_path)
    ids = [f"train_{i}" for i in range(100)]
    pd.DataFrame({"ID_code": ids, "var_0": np.arange(100), "target": np.tile([0, 1], 50)}).to_csv(
        tmp_path / "train.csv",
        index=False,
    )
    pd.DataFrame({"ID_code": ["test_0", "test_1"], "var_0": [4, 90]}).to_csv(
        tmp_path / "test.csv",
        index=False,
    )
    pd.DataFrame({"ID_code": ["test_1", "test_0"], "target": [0, 0]}).to_csv(
        tmp_path / "sample_submission.csv",
        index=False,
    )

    run(tmp_path, variant, variant, 42)
    store = RunStore(tmp_path)
    out = store.runs_dir / variant
    submission = pd.read_csv(out / "submission.csv")
    assert store.record(variant).metric == "roc_auc"
    assert submission["ID_code"].tolist() == ["test_1", "test_0"]
    np.testing.assert_allclose(submission["target"], np.load(out / "test.npy"))
    assert np.load(out / "oof.npy").shape == (100,)
    assert analyze(store, tmp_path, variant)["行数"].sum() == 100
