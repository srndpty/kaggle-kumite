import hashlib
import json
from pathlib import Path

import analyze
import numpy as np
import pytest
from dataset import Sample
from export import verify_selection, write_notebook
from submission import submission_images

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, files_sha256
from kumite.runs import RunStore


@pytest.fixture
def baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RunStore, list[dict[str, object]]]:
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    train_csv = data / "train.csv"
    train_csv.write_bytes(b"labels")
    samples = [Sample(data / f"{i}.jpg", str(i), [[0.0, 0.0, 10.0, 10.0]]) for i in range(3)]
    for sample in samples:
        sample.path.touch()
    monkeypatch.setattr(analyze, "load_samples", lambda path: (samples, {}))
    folds = analyze.split_images(samples, "source-v1")
    split = {sample.path.stem: fold for sample, fold in zip(samples, folds, strict=True)}
    rows: list[dict[str, object]] = [
        {
            "image_id": key,
            "source": key,
            "fold": fold,
            "boxes": samples[0].boxes,
            "scores": [0.9],
            "image_score": 1.0,
        }
        for key, fold in split.items()
    ]
    store = RunStore(tmp_path)
    for trial in ("E00", "E02", "E03", "E04", "E05"):
        record = ExperimentRecord(
            competition_id="074",
            run_id=trial,
            metric="wheat-image-precision",
            greater_is_better=True,
            data_version="test",
            seed=42,
            data_hash=files_sha256([train_csv, *(sample.path for sample in samples)]),
            split_version="source-v1",
            split_hash=array_sha256(np.asarray(folds, dtype=np.int64)),
            hypothesis="",
            changed_one_thing="",
            model="test",
            params={"trial": trial},
            fold_scores=[1.0] * 3,
            oof_score=1.0,
            code_version="test",
        )

        def write_outputs(output: Path) -> None:
            (output / "split.json").write_text(json.dumps(split), encoding="utf-8")
            (output / "oof.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        with store.reserve(trial) as reservation:
            store.publish(reservation, record, write_outputs)
    return store, rows


def test_export_rejects_altered_selection(
    baseline: tuple[RunStore, list[dict[str, object]]],
) -> None:
    store, _ = baseline
    choice = analyze.build_selection("E00", ["E02", "E03", "E04", "E05"])
    verify_selection("E00", choice, store)
    with pytest.raises(ValueError, match="candidate set"):
        verify_selection("E00", analyze.build_selection("E00", ["E02"]), store)
    choice["selected_run"] = "E02"
    with pytest.raises(ValueError, match="rescored OOF"):
        verify_selection("E02", choice, store)


@pytest.mark.parametrize(
    "mutation",
    ["log_score", "hash", "image_score", "nan", "score_range", "fold", "duplicate", "missing"],
)
def test_selection_rejects_inconsistent_evidence(
    baseline: tuple[RunStore, list[dict[str, object]]], mutation: str
) -> None:
    store, rows = baseline
    path = store.artifact_path("E00", "oof.jsonl")
    if mutation == "log_score":
        old = store.log_path.read_text(encoding="utf-8")
        store.log_path.write_text(
            old.replace('"oof_score": 1.0', '"oof_score": 0.5', 1), encoding="utf-8"
        )
    else:
        if mutation == "duplicate":
            rows.append(rows[0])
        elif mutation == "missing":
            rows.pop()
        elif mutation in {"image_score", "hash"}:
            rows[0]["image_score"] = 0.5
        elif mutation == "nan":
            rows[0]["boxes"] = [[float("nan"), 0, 10, 10]]
        elif mutation == "score_range":
            rows[0]["scores"] = [1.1]
        else:
            rows[0]["fold"] = (int(str(rows[0]["fold"])) + 1) % 3
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        old_hash = store.record("E00").artifacts["oof.jsonl"]
        path.write_bytes(payload.encode())
        if mutation != "hash":
            log = store.log_path.read_text(encoding="utf-8")
            store.log_path.write_text(
                log.replace(old_hash, hashlib.sha256(payload.encode()).hexdigest(), 1),
                encoding="utf-8",
            )
    with pytest.raises(ValueError):
        analyze.build_selection("E00", [])


def test_submission_ids_follow_template_and_reject_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    sample.write_bytes(b"image_id,PredictionString\nb,\na,\n")
    test = tmp_path / "test"
    test.mkdir()
    for key in ("a", "b"):
        (test / f"{key}.jpg").touch()
    assert [p.stem for p in submission_images(sample)] == ["b", "a"]
    (test / "extra.jpg").touch()
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="IDs"):
        submission_images(sample, hidden_template_sha256=digest)
    (test / "a.jpg").unlink()
    (test / "b.jpg").unlink()
    for key in ("d", "c"):
        (test / f"{key}.jpg").touch()
    with pytest.raises(ValueError, match="IDs"):
        submission_images(sample)
    assert [p.stem for p in submission_images(sample, hidden_template_sha256=digest)] == [
        "c",
        "d",
        "extra",
    ]


def test_cpu_notebook_writes_cpu_metadata_and_code(tmp_path: Path) -> None:
    write_notebook(
        tmp_path, "owner", "E02", "owner/weights", "hash", "template", 768, 0.5, False, "cpu"
    )
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["id"] == "owner/kumite-wheat-e02-cpu"
    notebook = json.loads((tmp_path / "submit.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    assert "torch.device('cpu')" in source
    assert "hidden_template_sha256='template'" in source
    compile(source, "wheat", "exec")
