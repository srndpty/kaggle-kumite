import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from inputs import official_data_hash
from predict import write_submission
from select_model import CANDIDATES, select
from train import score

from kumite.experiments import ExperimentRecord
from kumite.runs import RunStore


def test_cv_only_winner_requires_verified_submission(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    ids = np.arange(10)
    labels = ids % 2
    sample_ids = pd.Series([9, 2])
    pd.DataFrame({"id": ids, "is_duplicate": labels}).to_csv(data / "train.csv", index=False)
    pd.DataFrame({"test_id": sample_ids}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"test_id": sample_ids, "is_duplicate": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    roles = np.where(np.arange(5)[:, None] == ids // 2, 2, 1).astype(np.int8)
    data_hash = official_data_hash(tmp_path)
    test = np.array([0.25, 0.75])
    store = RunStore(tmp_path)

    def publish(run_id: str, experiment: str, cv_only: bool, oof: np.ndarray) -> None:
        def write(path: Path) -> None:
            np.save(path / "oof.npy", oof)
            np.savez(path / "split.npz", ids=ids, roles=roles)
            if not cv_only:
                np.save(path / "test.npy", test)
                probabilities = test[::-1] if run_id == "E02-bad" else test
                write_submission(sample_ids, probabilities, path / "submission.csv")

        with store.reserve(run_id) as reservation:
            store.publish(
                reservation,
                ExperimentRecord(
                    competition_id=tmp_path.name,
                    run_id=run_id,
                    metric="log_loss",
                    greater_is_better=False,
                    data_version="fixture",
                    data_hash=data_hash,
                    split_version="fixture-seed42",
                    split_hash=hashlib.sha256(roles.tobytes()).hexdigest(),
                    seed=42,
                    hypothesis="",
                    changed_one_thing="",
                    model="LR",
                    params={"experiment": experiment, "cv_only": cv_only},
                    fold_scores=[],
                    oof_score=score(labels, oof),
                    code_version="fixture",
                ),
                write,
            )

    winning_oof = np.where(labels, 0.9, 0.1)
    for run_id, experiment in CANDIDATES.items():
        cv_only = run_id == "E02-cv"
        publish(run_id, experiment, cv_only, winning_oof if cv_only else np.full(10, 0.5))
    with pytest.raises(ValueError, match=r"E02-cv.*提出成果物が未生成"):
        select(tmp_path, None)

    publish("E02-other", "E02", False, np.full(10, 0.4))
    with pytest.raises(ValueError, match="分割・OOF"):
        select(tmp_path, "E02-other")

    publish("E02-bad", "E02", False, winning_oof)
    with pytest.raises(ValueError, match="保存済みテスト確率"):
        select(tmp_path, "E02-bad")

    publish("E02-submit", "E02", False, winning_oof)
    result = select(tmp_path, "E02-submit")
    assert result["selected_run"] == "E02-submit"
    assert result["selection_evidence_run"] == "E02-cv"

    np.save(store.artifact_path("E04", "oof.npy"), np.full(10, 0.75))
    with pytest.raises(ValueError, match="ハッシュがログと一致しません"):
        select(tmp_path, "E02-submit")
