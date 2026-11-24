import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from export_checks import (
    CRITERION,
    source_hashes,
    verify_bundle,
    verify_selection,
    verify_sources,
)
from inference import submission
from matching import fit_text

from kumite.experiments import ExperimentRecord, append_record
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def record(run: str, score: float) -> ExperimentRecord:
    return ExperimentRecord(
        competition_id="037",
        run_id=run,
        metric="mean_f1",
        greater_is_better=True,
        data_version="titles",
        data_hash="data",
        split_version="split_v1",
        split_hash="fold",
        seed=42,
        hypothesis="",
        changed_one_thing="",
        model="",
        params={"trial": run},
        fold_scores=[score],
        oof_score=score,
        code_version="fixture",
    )


def test_selection_checks_requested_run_and_saved_scores(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    scores = {f"E{i:02}": i / 10 for i in range(5)}
    for run, score in scores.items():
        append_record(store.log_path, record(run, score))
    path = tmp_path / "selection.json"
    value = {
        "criterion": CRITERION,
        "metric": "mean_f1",
        "split_version": "split_v1",
        "oof_scores": scores,
        "selected_run": "E04",
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    verify_selection(store, "E04", path)
    with pytest.raises(ValueError, match="選定済み"):
        verify_selection(store, "E03", path)
    scores["E00"] = 0.9
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="OOF"):
        verify_selection(store, "E04", path)


def test_source_change_blocks_export(tmp_path: Path) -> None:
    for name in ("matching.py", "images.py", "inference.py"):
        (tmp_path / name).write_text("# cv", encoding="utf-8")
    hashes = source_hashes(tmp_path)
    historical = record("E04", 0.8)
    with pytest.raises(ValueError, match="未記録"):
        verify_sources(historical, hashes)
    saved = replace(
        historical, inputs={f"source/{name}": digest for name, digest in hashes.items()}
    )
    verify_sources(saved, hashes)
    (tmp_path / "matching.py").write_text("# changed", encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        verify_sources(saved, source_hashes(tmp_path))


@pytest.mark.parametrize("field", ["model", "vocabulary", "idf"])
def test_bundle_rejects_valid_json_with_modified_model_state(tmp_path: Path, field: str) -> None:
    config: dict[str, object] = {"model": "original", "vocabulary": {"abc": 0}, "idf": [1.0]}
    model = tmp_path / "model.json"
    model.write_text(json.dumps(config), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({model.name: file_sha256(model)}), encoding="utf-8")
    pinned = file_sha256(manifest)
    verify_bundle(tmp_path, pinned)
    config[field] = None
    model.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"model\.json"):
        verify_bundle(tmp_path, pinned)
    manifest.write_text(json.dumps({model.name: file_sha256(model)}), encoding="utf-8")
    with pytest.raises(ValueError, match="マニフェスト"):
        verify_bundle(tmp_path, pinned)


def test_single_posting_generates_self_match(tmp_path: Path) -> None:
    (encoder,) = fit_text(["same title"], words=False)
    config = {
        "trial": "E04",
        "model": None,
        "weights_sha256": None,
        "threshold": 0.5,
        "text": [
            {
                "analyzer": "char",
                "ngram_range": [3, 5],
                "vocabulary": encoder.vocabulary_,
                "idf": encoder.idf_.tolist(),
            }
        ],
    }
    (tmp_path / "model.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame({"posting_id": ["test_1"], "title": ["same title"], "image_phash": ["00"]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    assert submission(tmp_path, tmp_path).to_dict("records") == [
        {"posting_id": "test_1", "matches": "test_1"}
    ]
