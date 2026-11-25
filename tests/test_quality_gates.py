import csv
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kumite.crossfit import Split, split_roles
from kumite.experiments import ExperimentRecord, compare_runs
from kumite.provenance import array_sha256, file_sha256
from kumite.quality_gates import (
    Artifact,
    Scores,
    SelectionPlan,
    SourceSpec,
    dense_oof,
    publish_export,
    publish_gated_cv,
    reserve_cv,
    select_runs,
    selection_plan_hash,
    verify_cv_run,
    verify_dependencies,
    verify_export,
    verify_selection,
    write_selection,
)
from kumite.runs import QUALITY_GATE_MARKER, RunStore

Y = np.array([5.0, 5.0, 5.0])
SPLITS: list[Split] = [(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([2]))]
ASSESS = dense_oof(
    Y,
    SPLITS,
    lambda y, pred: float(np.abs(y - pred).mean()),
    expected_shape=(3,),
    data_hash="train",
)


@pytest.fixture
def source(tmp_path: Path) -> SourceSpec:
    (tmp_path / "model.py").write_text("model", encoding="utf-8")
    return SourceSpec(tmp_path, ("model.py",), {"features": ["a"], "seed": 42})


def _record(
    run_id: str, error: float = 0.0, inputs: Mapping[str, str] | None = None
) -> ExperimentRecord:
    return ExperimentRecord(
        competition_id="test",
        run_id=run_id,
        metric="mae",
        greater_is_better=False,
        data_version="raw",
        data_hash="train",
        split_version="ts",
        split_hash=array_sha256(split_roles(len(Y), SPLITS)),
        seed=42,
        hypothesis="仮説",
        changed_one_thing="なし",
        model="test",
        code_version="test",
        fold_scores=[error, error],
        oof_score=error,
        inputs=dict(inputs or {}),
    )


def _publish(
    store: RunStore,
    source: SourceSpec,
    record: ExperimentRecord,
    oof: np.ndarray | None = None,
    *,
    plan: SelectionPlan | None = None,
) -> Path:
    if oof is None:
        oof = np.array([np.nan, 5 + record.oof_score, 5 + record.oof_score])
    with reserve_cv(store, record.run_id, source, record.inputs, plan) as (reservation, snapshot):
        return publish_gated_cv(
            store,
            reservation,
            record,
            lambda path: np.save(path / "oof.npy", oof),
            source=source,
            before_training=snapshot,
            assess=ASSESS,
        )


@pytest.mark.parametrize(
    ("oof", "message"),
    [
        ([0, 5, 5], "未検証行"),
        ([np.nan, np.nan, 5], "検証行"),
        ([np.nan, 5, np.inf], "検証行"),
        ([np.nan, 5], "OOF の形"),
        ([np.nan, 4, 6], "再計算値"),
    ],
)
def test_invalid_oof_prevents_publication(
    tmp_path: Path, source: SourceSpec, oof: list[float], message: str
) -> None:
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match=message):
        _publish(store, source, _record("E00"), np.array(oof))
    assert store.records() == []
    assert store.orphans() == []


@pytest.mark.parametrize(
    ("field", "message"), [("data_hash", "評価データ"), ("split_hash", "検証分割")]
)
def test_oof_requires_expected_evaluation_identity(
    tmp_path: Path, source: SourceSpec, field: str, message: str
) -> None:
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match=message):
        _publish(store, source, replace(_record("E00"), **{field: "wrong"}))
    assert store.records() == []


@pytest.mark.parametrize("change", ["missing", "changed", "rehashed"])
def test_selection_checks_even_losing_candidate(
    tmp_path: Path, source: SourceSpec, change: str
) -> None:
    store = RunStore(tmp_path)
    _publish(store, source, _record("E00"))
    path = _publish(store, source, _record("E01", 1.0)) / "oof.npy"
    plan = SelectionPlan(("E00", "E01"))
    assert select_runs(store, plan, ASSESS).winner == "E00"
    if change == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            select_runs(store, plan, ASSESS)
    else:
        np.save(path, np.array([np.nan, 4.0, 4.0]))  # 同じ MAE
        if change == "rehashed":
            log = store.log_path.read_text(encoding="utf-8")
            old = store.record("E01").artifacts["oof.npy"]
            store.log_path.write_text(log.replace(old, file_sha256(path)), encoding="utf-8")
        with pytest.raises(ValueError, match=r"ハッシュ|証跡"):
            select_runs(store, plan, ASSESS)


def test_selection_plan_hash_is_fixed_at_training_start(tmp_path: Path, source: SourceSpec) -> None:
    store = RunStore(tmp_path)
    plan = SelectionPlan(("E00", "E01"))
    _publish(store, source, _record("E00"), plan=plan)
    _publish(store, source, _record("E01", 1.0), plan=plan)
    assert select_runs(store, plan, ASSESS).winner == "E00"
    evidence = json.loads(
        store.artifact_path("E00", "cv_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["plan_hash"] == selection_plan_hash(plan)
    with pytest.raises(ValueError, match="一致しません"):
        select_runs(store, SelectionPlan(("E01", "E00")), ASSESS)
    with (
        pytest.raises(ValueError, match="含まれていません"),
        reserve_cv(store, "E02", source, {}, SelectionPlan(("E00",))),
    ):
        pass
    assert [record.run_id for record in store.records()] == ["E00", "E01"]
    assert store.orphans() == []


def test_legacy_run_cannot_enter_strict_selection(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    with store.reserve("E00") as reservation:
        store.publish(reservation, _record("E00"), lambda _: None)
    with pytest.raises(LookupError, match="cv_evidence"):
        verify_cv_run(store, "E00", ASSESS)


def test_enabled_store_requires_gate(tmp_path: Path, source: SourceSpec) -> None:
    (tmp_path / QUALITY_GATE_MARKER).touch()
    store = RunStore(tmp_path)
    with store.reserve("E00") as reservation, pytest.raises(ValueError, match="publish_gated_cv"):
        store.publish(reservation, _record("E00"), lambda _: pytest.fail("書き込んではいけない"))
    assert store.records() == []
    _publish(store, source, _record("E00"))
    assert verify_cv_run(store, "E00", ASSESS).oof_score == 0


def test_dependency_checks_transitive_bytes_and_declared_digest(
    tmp_path: Path, source: SourceSpec
) -> None:
    store = RunStore(tmp_path)
    parent = _publish(store, source, _record("E00"))
    inputs = store.input_hashes("E00", "oof.npy")
    with pytest.raises(ValueError, match="親の記録"):
        _publish(store, source, _record("bad", inputs={"E00/oof.npy": "wrong"}))
    with pytest.raises(ValueError, match="循環"):
        verify_dependencies(store, inputs, run_id="E00")
    _publish(store, source, _record("E01", inputs=inputs))
    child_inputs = store.input_hashes("E01", "oof.npy")
    original = store.log_path.read_bytes()
    entries = [json.loads(line) for line in original.splitlines()]
    entries[1]["inputs"] = {}
    store.log_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="証跡"):
        verify_dependencies(store, child_inputs, run_id="E02")
    store.log_path.write_bytes(original)
    (parent / "oof.npy").unlink()
    with pytest.raises(FileNotFoundError), reserve_cv(store, "E02", source, child_inputs):
        pytest.fail("学習を開始してはいけない")


def test_training_source_change_prevents_publication(tmp_path: Path, source: SourceSpec) -> None:
    store = RunStore(tmp_path)
    with reserve_cv(store, "E00", source, {}) as (reservation, snapshot):
        (tmp_path / "model.py").write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="学習開始時"):
            publish_gated_cv(
                store,
                reservation,
                _record("E00"),
                lambda path: np.save(path / "oof.npy", np.array([np.nan, 5, 5])),
                source=source,
                before_training=snapshot,
                assess=ASSESS,
            )
    assert store.records() == []


@pytest.mark.parametrize(
    "mutation",
    [
        "winner",
        "candidates",
        "score",
        "source",
        "settings",
        "source_during",
        "output",
        "duplicate_id",
        "missing_id",
        "nan_prediction",
    ],
)
def test_export_rejects_invalid_evidence(tmp_path: Path, source: SourceSpec, mutation: str) -> None:
    store = RunStore(tmp_path)
    _publish(store, source, _record("E00", 1.0))
    _publish(store, source, _record("E01"))
    plan = SelectionPlan(("E00", "E01"), tolerance=1.0)
    receipt = tmp_path / "selection.json"
    selected = select_runs(store, plan, ASSESS)
    assert selected.winner == "E00"
    write_selection(receipt, selected)
    assert verify_selection(store, plan, receipt, ASSESS) == selected

    if mutation in {"winner", "candidates", "score"}:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if mutation == "winner":
            payload["winner"] = "E01"
        elif mutation == "candidates":
            payload["plan"]["candidates"].pop()
        else:
            payload["scores"]["E01"] = 2.0
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "source":
        (tmp_path / "model.py").write_text("changed", encoding="utf-8")
    elif mutation == "settings":
        source = replace(source, settings={"features": ["b"], "seed": 42})

    called: list[str] = []
    invalid_csv = {
        "duplicate_id": "id,target\n1,5\n1,6\n",
        "missing_id": "id,target\n1,5\n",
        "nan_prediction": "id,target\n1,5\n2,NaN\n",
    }

    def write(path: Path, winner: ExperimentRecord) -> None:
        called.append(winner.run_id)
        if mutation != "output":
            (path / "submission.csv").write_text(
                invalid_csv.get(mutation, "id,target\n1,5\n2,6\n"), encoding="utf-8"
            )
        if mutation == "source_during":
            (tmp_path / "model.py").write_text("changed", encoding="utf-8")

    def validate_outputs(path: Path) -> None:
        with (path / "submission.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "target"]:
                raise ValueError("提出スキーマ")
            rows = list(reader)
        if [row["id"] for row in rows] != ["1", "2"]:
            raise ValueError("提出 ID")
        if not all(math.isfinite(float(row["target"])) for row in rows):
            raise ValueError("提出予測値")

    def export() -> Path:
        return publish_export(
            store,
            "final",
            plan=plan,
            receipt=receipt,
            assess=ASSESS,
            source=source,
            write=write,
            required_outputs=("submission.csv",),
            validate_outputs=validate_outputs,
        )

    original = store.log_path.read_bytes()
    with pytest.raises(ValueError, match=r"選定記録|ソースまたは設定|必須|提出 ID|提出予測値"):
        export()
    assert called == (["E00"] if mutation in {"output", "source_during", *invalid_csv} else [])
    export_store = RunStore(tmp_path / "outputs" / "exports")
    assert export_store.records() == []
    assert store.log_path.read_bytes() == original

    (tmp_path / "model.py").write_text("model", encoding="utf-8")
    source = replace(source, settings={"features": ["a"], "seed": 42})
    receipt.unlink()
    # 確定済みの選定記録がない export は、学習も公開もせず、記録も作らない
    writes = list(called)
    with pytest.raises(FileNotFoundError):
        export()
    assert called == writes and export_store.records() == [] and not receipt.exists()
    write_selection(receipt, selected)
    with pytest.raises(FileExistsError):
        write_selection(receipt, selected)
    if mutation in {"output", "source_during", *invalid_csv}:
        mutation = "restored"
    final = export()
    assert (final / "submission.csv").is_file()
    assert (final / "selection.json").read_bytes() == receipt.read_bytes()
    assert store.log_path.read_bytes() == original

    exported = verify_export(export_store, "final")
    assert exported.inputs == {
        f"cv:{candidate}/{name}": digest
        for candidate in plan.candidates
        for name, digest in store.record(candidate).artifacts.items()
    }
    assert exported.metric == "cv_reference:mae"
    assert exported.oof_score == store.record("E00").oof_score
    assert exported.late_score is None
    with pytest.raises(ValueError, match="評価条件"):
        compare_runs(store.record("E00"), exported)

    losing_oof = store.artifact_path("E01", "oof.npy")
    np.save(losing_oof, np.array([np.nan, 7.0, 7.0]))
    with pytest.raises(ValueError, match="ハッシュ"):
        verify_export(export_store, "final")
    losing_oof.unlink()
    with pytest.raises(FileNotFoundError):
        verify_export(export_store, "final")


def test_source_audit_checks_local_and_relative_imports(tmp_path: Path) -> None:
    src = tmp_path / "src"
    package = src / "model"
    package.mkdir(parents=True)
    (src / "train.py").write_text("from model import features\nimport numpy\n", encoding="utf-8")
    (package / "__init__.py").touch()
    (package / "features.py").write_text("from . import transforms\n", encoding="utf-8")
    (package / "transforms.py").touch()
    source = SourceSpec(
        tmp_path, ("src/train.py", "src/model/__init__.py"), {}, import_roots=("src",)
    )
    with pytest.raises(ValueError, match=r"未宣言.*features\.py"):
        source.snapshot()
    source = replace(source, files=(*source.files, "src/model/features.py"))
    with pytest.raises(ValueError, match=r"未宣言.*transforms\.py"):
        source.snapshot()
    source = replace(source, files=(*source.files, "src/model/transforms.py"))
    assert source.snapshot()


def test_custom_oof_can_verify_keyed_image_records(tmp_path: Path, source: SourceSpec) -> None:
    store = RunStore(tmp_path)
    expected = {"image-a", "image-b"}
    record = _record("image")

    def assess(artifact: Artifact, _: ExperimentRecord) -> Scores:
        rows = artifact("detections.txt").read_text(encoding="utf-8").splitlines()
        if len(rows) != len(expected) or set(rows) != expected:
            raise ValueError("画像 ID の欠落または重複")
        return [0.0, 0.0], 0.0

    for text in ("image-a\nimage-a\n", "image-a\nimage-b\n"):

        def write(path: Path, content: str = text) -> None:
            (path / "detections.txt").write_text(content, encoding="utf-8")

        with reserve_cv(store, "image", source, {}) as (reservation, snapshot):
            if text.count("image-a") == 2:
                with pytest.raises(ValueError, match="画像 ID"):
                    publish_gated_cv(
                        store,
                        reservation,
                        record,
                        write,
                        source=source,
                        before_training=snapshot,
                        assess=assess,
                    )
            else:
                publish_gated_cv(
                    store,
                    reservation,
                    record,
                    write,
                    source=source,
                    before_training=snapshot,
                    assess=assess,
                )
    assert verify_cv_run(store, "image", assess).run_id == "image"
