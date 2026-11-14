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
    data_version="v1",
    split_version="skf5-seed42",
    seed=42,
    hypothesis="基準",
    changed_one_thing="なし",
    model="lightgbm",
    params={"num_leaves": 31},
    fold_scores=[0.52, 0.51],
    oof_score=0.515,
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


def test_compare_runs_pairs_folds() -> None:
    comparison = compare_runs(BASELINE, IMPROVED)

    assert comparison.diffs == pytest.approx([-0.01, 0.005])
    assert comparison.mean_diff == pytest.approx(-0.0025)
    assert (comparison.n_lower, comparison.n_higher) == (1, 1)


def test_compare_runs_rejects_different_splits() -> None:
    with pytest.raises(ValueError, match="分割"):
        compare_runs(BASELINE, replace(IMPROVED, split_version="kf5-seed0"))
