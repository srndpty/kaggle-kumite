from dataclasses import replace
from pathlib import Path

import pytest

from kumite.experiments import (
    LOG_FILE_NAME,
    ExperimentRecord,
    append_record,
    compare_runs,
    find_record,
    read_records,
)

BASELINE = ExperimentRecord(
    competition_id="001",
    run_id="E00",
    metric="multi_logloss",
    greater_is_better=False,
    data_version="v1",
    data_hash="d" * 64,
    split_version="skf5-seed42",
    split_hash="s" * 64,
    seed=42,
    hypothesis="基準",
    changed_one_thing="なし",
    model="lightgbm",
    params={"num_leaves": 31},
    fold_scores=[0.52, 0.51],
    oof_score=0.515,
    code_version="abc",
)
IMPROVED = replace(
    BASELINE,
    run_id="E01",
    hypothesis="正則化で過学習が減る",
    changed_one_thing="lambda_l2=1.0",
    params={},
    fold_scores=[0.51, 0.515],
    oof_score=0.5125,
    late_score=0.507,
    conclusion="改善候補",
)


def test_records_round_trip_in_order(tmp_path: Path) -> None:
    log_path = tmp_path / LOG_FILE_NAME

    append_record(log_path, BASELINE)
    append_record(log_path, IMPROVED)

    records = read_records(log_path)
    assert records == [BASELINE, IMPROVED]
    assert find_record(records, "E01") == IMPROVED
    with pytest.raises(ValueError, match="E00"):
        append_record(log_path, BASELINE)


def test_records_keep_full_precision(tmp_path: Path) -> None:
    log_path = tmp_path / LOG_FILE_NAME
    precise = replace(BASELINE, fold_scores=[0.123456789012345, 0.1], oof_score=0.111111111111)

    append_record(log_path, precise)

    assert read_records(log_path)[0].fold_scores[0] == 0.123456789012345


def test_compare_runs_pairs_folds() -> None:
    comparison = compare_runs(BASELINE, IMPROVED)

    assert comparison.diffs == pytest.approx([-0.01, 0.005])
    assert comparison.mean_diff == pytest.approx(-0.0025)
    assert (comparison.n_improved, comparison.n_worsened) == (1, 1)


def test_compare_runs_counts_improvement_by_metric_direction() -> None:
    base = replace(BASELINE, greater_is_better=True)
    other = replace(IMPROVED, greater_is_better=True, fold_scores=[0.53, 0.52])

    assert compare_runs(base, other).n_improved == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("competition_id", "002"),
        ("metric", "accuracy"),
        ("greater_is_better", True),
        ("data_hash", "x" * 64),
        ("split_version", "kf5-seed0"),
        ("split_hash", "y" * 64),
    ],
)
def test_compare_runs_rejects_different_conditions(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        compare_runs(BASELINE, replace(IMPROVED, **{field: value}))
