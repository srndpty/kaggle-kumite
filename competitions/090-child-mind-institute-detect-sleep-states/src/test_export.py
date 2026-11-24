"""選定記録と提出物の照合。"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from export import validate_final
from pipeline import read_model_bundle
from select_model import validate_candidates, validate_main_run
from train import ROOT

from kumite.experiments import ExperimentRecord


def record() -> ExperimentRecord:
    return ExperimentRecord(
        competition_id=ROOT.name,
        run_id="E04",
        metric="event_detection_ap",
        greater_is_better=True,
        data_version="official-1min",
        data_hash="data",
        split_version="series-3fold-v1",
        split_hash="split",
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="LightGBM",
        fold_scores=[0.1] * 3,
        oof_score=0.1,
        code_version="test",
    )


def test_final_requires_selected_run_and_matching_hashes(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    payload = {"selected_run": "E04", "data_hash": "data", "split_hash": "split"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_final(record(), path) == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="selected run"):
        validate_final(replace(record(), run_id="E00"), path)
    with pytest.raises(ValueError, match="incompatible"):
        validate_final(replace(record(), split_hash="different"), path)


def test_selection_rejects_invalid_metric_seed_and_candidates() -> None:
    original = record()
    for changed in (replace(original, metric="accuracy"), replace(original, seed=2026)):
        with pytest.raises(ValueError, match="incompatible"):
            validate_main_run(changed, "data", "split")
    runs = ["E00", "E01", "E02", "E03", "E04"]
    validate_candidates(runs)
    for invalid in (runs[1:], [*runs, "E04"], [*runs[:-1], "E04-seed2026"]):
        with pytest.raises(ValueError, match="exactly once"):
            validate_candidates(invalid)


def test_modified_model_bundle_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    payload = b'{"onset": ["model"]}'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert read_model_bundle(path, expected) == {"onset": ["model"]}
    path.write_bytes(b"bad json")
    with pytest.raises(ValueError, match="SHA-256"):
        read_model_bundle(path, expected)
