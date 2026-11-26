import json
from pathlib import Path

import numpy as np
import pytest
from run import fit, fold_prediction, mae, predict, validate_export_outputs, validate_submission
from windows import fold_ids, rolling_std_quantile, verify_manifest, windows

from kumite.checks import check_label_independence, perturb_targets
from kumite.crossfit import folds_to_splits
from kumite.provenance import array_sha256


def test_windows_keep_complete_cycles_and_use_end_labels() -> None:
    y = np.r_[2.0, 1.0, np.linspace(20, 0, 22), np.linspace(20, 0, 12)]
    x = np.arange(len(y), dtype=np.int16)
    chunks = [(x[:2], y[:2]), (x[2:8], y[2:8]), (x[8:24], y[8:24]), (x[24:], y[24:])]
    rows, values, boundaries = windows(chunks, size=10)

    np.testing.assert_array_equal(boundaries[:, 0], [2, 24])
    np.testing.assert_array_equal(rows[:, :3], [[2, 12, 1], [12, 22, 1]])
    np.testing.assert_array_equal(rows[:, 3], y[[11, 21]])
    np.testing.assert_array_equal(values[:, 0], [6.5, 16.5])


@pytest.mark.parametrize("width", [1, 7, 48, 49, 50, 51, 150_000, 1_000_000])
def test_windows_are_invariant_to_chunk_width(width: int) -> None:
    y = np.concatenate([np.linspace(9, 0, n) for n in (49, 117, 77, 131, 71)])
    x = np.random.default_rng(42).integers(-30, 31, size=len(y), dtype=np.int16)
    expected = windows([(x, y)], size=50)
    actual = windows(
        ((x[i : i + width], y[i : i + width]) for i in range(0, len(y), width)),
        size=50,
    )
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    ("key", "column", "name"),
    [
        ("rows", 0, "positions"),
        ("rows", 3, "labels"),
        ("train", 0, "train"),
        ("test", 0, "test"),
    ],
)
def test_manifest_rejects_finite_changes(tmp_path: Path, key: str, column: int, name: str) -> None:
    arrays = {
        "rows": np.array([[0.0, 50.0, 1.0, 2.5]]),
        "train": np.array([[1.0]]),
        "test": np.array([[2.0]]),
        "folds": np.array([0]),
        "ids": np.array(["a"]),
        "boundaries": np.array([[0.0, 0.0, 8.0]]),
    }
    manifest_arrays = {k: v for k, v in arrays.items() if k != "rows"}
    manifest_arrays["positions"] = np.array([[0, 50, 1]], dtype=np.int64)
    manifest_arrays["labels"] = np.array([2.5])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "official_hash": "official",
                "arrays": {k: array_sha256(v) for k, v in manifest_arrays.items()},
            }
        ),
        encoding="utf-8",
    )
    verify_manifest(manifest, "official", arrays)
    arrays[key][0, column] += 1.0
    with pytest.raises(ValueError, match=name):
        verify_manifest(manifest, "official", arrays)


def test_export_predictions_match_saved_booster(tmp_path: Path) -> None:
    x = np.arange(80, dtype=float).reshape(-1, 1)
    model = fit(x, x.ravel()).booster_
    model.save_model(str(tmp_path / "model.txt"))
    test = x[[15, 65]]
    first, second = predict(model, test)
    submission = tmp_path / "submission.csv"
    ids = np.array(["a", "b"])
    submission.write_text(f"seg_id,time_to_failure\na,{first}\nb,{second}\n", encoding="utf-8")
    validate_export_outputs(tmp_path, ids, test)
    submission.write_text(f"seg_id,time_to_failure\na,{first + 1}\nb,{second}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="再推論"):
        validate_export_outputs(tmp_path, ids, test)


def test_fold_isolates_whole_cycles() -> None:
    cycles = np.repeat(np.arange(11), 3)
    folds = fold_ids(cycles)
    np.testing.assert_array_equal(folds[::3], [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    for train, valid in folds_to_splits(folds):
        assert not set(cycles[train]) & set(cycles[valid])


def test_rolling_std_quantile_matches_direct_computation() -> None:
    x = np.array([2.0, 5.0, 8.0, 8.0, 11.0, 13.0])
    width = 3
    expected = np.quantile([x[i : i + width].std() for i in range(len(x) - width + 1)], 0.20)
    assert rolling_std_quantile(x, width) == pytest.approx(expected)


def test_fold_fit_is_independent_of_validation_targets() -> None:
    x = np.arange(80, dtype=float).reshape(-1, 1)
    y = x.ravel() / 10
    train, valid = np.arange(60), np.arange(60, 80)
    changed = np.arange(len(y)) >= 60
    check_label_independence(
        lambda targets: (fold_prediction(x, targets, train, valid),),
        y,
        perturb_targets(y, changed),
    )
    assert mae(np.array([0.0, 1.0]), np.array([1.0, 4.0])) == 2.0


@pytest.mark.parametrize("rows", ["a,0\n", "a,0\na,0\n", "b,0\na,0\n", "a,0\nb,nan\n"])
def test_submission_rejects_invalid_rows(tmp_path: Path, rows: str) -> None:
    path = tmp_path / "submission.csv"
    path.write_text("seg_id,time_to_failure\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission(path, np.array(["a", "b"]))
