import json
from pathlib import Path

import numpy as np

from kumite.crossfit import CrossFitResult, Split
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.runs import RunStore


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


EVALUATION = Evaluation(
    competition_id="019",
    metric="mae",
    greater_is_better=False,
    score=_mae,
    data_hash="d" * 64,
    split_version="ts2",
    split_hash="s" * 64,
    seed=42,
)
TRIAL = Trial(hypothesis="基準", changed_one_thing="なし", model="median", data_version="raw")
Y = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
# 行 0・1 は検証されない（OOF は NaN）。
SPLITS: list[Split] = [
    (np.array([0, 1]), np.array([2, 3])),
    (np.array([0, 1, 2, 3]), np.array([4, 5])),
]
RESULT = CrossFitResult(
    oof=np.array([np.nan, np.nan, 11.0, 11.0, 13.0, 13.0]),
    query=np.array([12.0]),
    infos=[{"n": 2}, {"n": 4}],
)


def _write_submission(out_dir: Path) -> None:
    (out_dir / "submission.csv").write_text("Id,Sales\n1,12.0\n", encoding="utf-8")


def test_publish_cv_run_scores_validated_rows_only(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    with store.reserve("E00") as reservation:
        final = publish_cv_run(
            store, reservation, EVALUATION, TRIAL, Y, SPLITS, RESULT, 1.5, _write_submission
        )

    record = store.record("E00")
    assert record.fold_scores == [1.0, 3.0]
    assert record.oof_score == 2.0  # NaN の行（未検証）は含めない
    assert record.competition_id == "019"
    assert record.split_version == "ts2"
    assert record.runtime_sec == 1.5
    assert set(record.artifacts) == {"oof.npy", "test.npy", "fold_info.json", "submission.csv"}
    np.testing.assert_array_equal(store.load_array("E00", "oof.npy"), RESULT.oof)
    assert json.loads((final / "fold_info.json").read_text(encoding="utf-8")) == RESULT.infos
