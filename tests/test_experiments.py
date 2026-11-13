from pathlib import Path

from kumite.experiments import LOG_FILE_NAME, ExperimentRecord, append_record, read_records


def test_records_round_trip_in_order(tmp_path: Path) -> None:
    log_path = tmp_path / LOG_FILE_NAME
    baseline = ExperimentRecord(
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
    improved = ExperimentRecord(
        competition_id="001",
        run_id="E01",
        data_version="v1",
        split_version="skf5-seed42",
        seed=42,
        hypothesis="正則化で過学習が減る",
        changed_one_thing="lambda_l2=1.0",
        model="lightgbm",
        fold_scores=[0.51, 0.50],
        oof_score=0.505,
        late_score=0.507,
        conclusion="改善候補",
    )

    append_record(log_path, baseline)
    append_record(log_path, improved)

    assert read_records(log_path) == [baseline, improved]
