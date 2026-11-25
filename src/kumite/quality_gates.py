"""新規 run の公開・選定・export で、保存した証跡を検証する。"""

import json
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np

from kumite.crossfit import Split, split_roles
from kumite.experiments import ExperimentRecord, compare_runs
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, code_version, file_sha256
from kumite.runs import CV_EVIDENCE, QUALITY_GATE_MARKER, Reservation, RunStore, validate_run_id
from kumite.source_audit import verify_local_imports

SELECTION = "selection.json"
EXPORT_EVIDENCE = "export_evidence.json"

Artifact = Callable[[str], Path]
Scores = tuple[list[float], float]
AssessOOF = Callable[[Artifact, ExperimentRecord], Scores]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class SourceSpec:
    """CV と export で共有するファイルと実効設定。パスは root からの相対 POSIX 形式。"""

    root: Path
    files: tuple[str, ...]
    settings: Mapping[str, object]
    import_roots: tuple[str, ...] = (".",)

    def snapshot(self) -> str:
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError("ソースファイルは重複なく 1 件以上必要です")
        hashes: dict[str, str] = {}
        root = self.root.resolve()
        for name in self.files:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() != name:
                raise ValueError(f"ソースパスが root からの相対パスではありません: {name}")
            hashes[name] = file_sha256(path)
        verify_local_imports(root, set(hashes), self.import_roots)
        snapshot: dict[str, object] = {
            "version": 1,
            "files": hashes,
            "settings": dict(self.settings),
        }
        if self.import_roots != (".",):
            snapshot["import_roots"] = self.import_roots
        return _json(snapshot)


def verify_dependencies(
    store: RunStore, inputs: Mapping[str, str], *, run_id: str | None = None
) -> None:
    """宣言された入力を、親の記録と実ファイル、その親の依存先まで照合する。"""
    active = {run_id} if run_id is not None else set[str]()
    verified: set[str] = set()
    strict = (store.competition_dir / QUALITY_GATE_MARKER).is_file()

    def visit(dependencies: Mapping[str, str]) -> None:
        for reference, digest in dependencies.items():
            parent, sep, name = reference.partition("/")
            validate_run_id(parent)
            if not sep or not name or name in {".", ".."} or any(c in name for c in "/\\:"):
                raise ValueError(f"依存成果物の参照が不正です: {reference}")
            if parent in active:
                raise ValueError(f"run の依存が循環しています: {reference}")
            record = store.record(parent)
            if record.artifacts.get(name) != digest:
                raise ValueError(f"依存成果物のハッシュが親の記録と一致しません: {reference}")
            store.artifact_path(parent, name)
            if parent not in verified:
                if strict or CV_EVIDENCE in record.artifacts:
                    _read_evidence(store, record)
                active.add(parent)
                visit(record.inputs)
                active.remove(parent)
                verified.add(parent)

    visit(inputs)


def _verify_scores(record: ExperimentRecord, actual: Scores) -> None:
    folds, overall = actual
    if not folds or len(folds) != len(record.fold_scores):
        raise ValueError(f"{record.run_id}: 再計算した fold 数が記録と一致しません")
    for expected, score in zip(
        [*record.fold_scores, record.oof_score], [*folds, overall], strict=True
    ):
        if not math.isfinite(score) or not math.isclose(expected, score, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"{record.run_id}: OOF スコアの再計算値が記録と一致しません")


def _record_identity(record: ExperimentRecord) -> dict[str, object]:
    evidence = asdict(record)
    for name in ("created_at", "artifacts", "late_score", "conclusion", "next_experiment"):
        del evidence[name]
    return evidence


def _read_evidence(
    store: RunStore, record: ExperimentRecord, name: str = CV_EVIDENCE
) -> tuple[Path, str]:
    evidence = store.artifact_path(record.run_id, name)
    decoded: object = json.loads(evidence.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{record.run_id}: {name} の形式が不正です")
    payload = cast(dict[str, object], decoded)
    manifest = {key: digest for key, digest in record.artifacts.items() if key != name}
    source = payload.get("source")
    if (
        not manifest
        or not isinstance(source, str)
        or _json(payload)
        != _json({"source": source, "record": _record_identity(record), "artifacts": manifest})
    ):
        raise ValueError(f"{record.run_id}: {name} の証跡と実験ログが一致しません")
    return evidence, source


@dataclass(frozen=True)
class CVStart:
    reservation: Reservation
    source: str
    inputs: tuple[tuple[str, str], ...]


@contextmanager
def reserve_cv(
    store: RunStore, run_id: str, source: SourceSpec, inputs: Mapping[str, str]
) -> Generator[tuple[Reservation, CVStart]]:
    """学習前に ID を予約し、依存先を検査してソースと設定を採取する。"""
    with store.reserve(run_id) as reservation:
        verify_dependencies(store, inputs, run_id=run_id)
        yield reservation, CVStart(reservation, source.snapshot(), tuple(sorted(inputs.items())))


def publish_gated_cv(
    store: RunStore,
    reservation: Reservation,
    record: ExperimentRecord,
    write: Callable[[Path], None],
    *,
    source: SourceSpec,
    before_training: CVStart,
    assess: AssessOOF,
) -> Path:
    """一時ディレクトリ内の OOF を再採点し、合格した run だけを公開する。"""
    if before_training.reservation != reservation or before_training.inputs != tuple(
        sorted(record.inputs.items())
    ):
        raise ValueError("CV の予約または依存宣言が学習開始時と一致しません")
    verify_dependencies(store, record.inputs, run_id=record.run_id)

    def checked_write(path: Path) -> None:
        write(path)
        artifacts = {item.name: item for item in path.iterdir()}
        if not artifacts:
            raise ValueError("CV 成果物がありません")
        _verify_scores(record, assess(artifacts.__getitem__, record))
        verify_dependencies(store, record.inputs, run_id=record.run_id)

    def seal(artifacts: Mapping[str, str]) -> str:
        if source.snapshot() != before_training.source:
            raise ValueError("学習開始時からソースまたは設定が変わりました")
        return (
            _json(
                {
                    "source": before_training.source,
                    "record": _record_identity(record),
                    "artifacts": dict(artifacts),
                }
            )
            + "\n"
        )

    return store.publish(reservation, record, checked_write, seal=(CV_EVIDENCE, seal))


def verify_cv_run(store: RunStore, run_id: str, assess: AssessOOF) -> ExperimentRecord:
    """証跡・全成果物・依存先・再計算スコアを照合する。"""
    record, _ = _verify_cv_run(store, run_id, assess)
    return record


def _verify_cv_run(
    store: RunStore, run_id: str, assess: AssessOOF
) -> tuple[ExperimentRecord, float]:
    validate_run_id(run_id)
    record = store.record(run_id)
    evidence, _ = _read_evidence(store, record)
    paths = {
        name: store.artifact_path(run_id, name) for name in record.artifacts if name != CV_EVIDENCE
    }
    if {item.name for item in evidence.parent.iterdir()} != set(record.artifacts):
        raise ValueError(f"{run_id}: 成果物の一覧が記録と一致しません")
    verify_dependencies(store, record.inputs, run_id=run_id)
    scores = assess(paths.__getitem__, record)
    _verify_scores(record, scores)
    return record, scores[1]


def dense_oof(
    y: np.ndarray,
    splits: Sequence[Split],
    score: Callable[[np.ndarray, np.ndarray], float],
    *,
    expected_shape: tuple[int, ...],
    data_hash: str,
) -> AssessOOF:
    """行順の OOF 用。時系列分割では検証対象外の全要素が NaN であることを求める。"""
    roles = split_roles(len(y), splits)
    split_hash = array_sha256(roles)
    valid = (roles == 2).any(axis=0)
    folds = [idx.copy() for _, idx in splits]

    def assess(artifact: Artifact, record: ExperimentRecord) -> Scores:
        if record.data_hash != data_hash:
            raise ValueError("評価データのハッシュが記録と一致しません")
        if record.split_hash != split_hash:
            raise ValueError("検証分割のハッシュが記録と一致しません")
        oof = np.load(artifact("oof.npy"), allow_pickle=False)
        if oof.shape != expected_shape or oof.ndim == 0 or len(oof) != len(y):
            raise ValueError("OOF の形が評価契約と一致しません")
        if not np.isfinite(oof[valid]).all() or not np.isnan(oof[~valid]).all():
            raise ValueError("OOF の検証行は有限値、未検証行は NaN が必要です")
        return [score(y[idx], oof[idx]) for idx in folds], score(y[valid], oof[valid])

    return assess


@dataclass(frozen=True)
class SelectionPlan:
    """事前に固定する候補の全件と同点判定。許容差内では候補順を優先する。"""

    candidates: tuple[str, ...]
    tolerance: float = 0.0


@dataclass(frozen=True)
class Selection:
    plan: SelectionPlan
    winner: str
    scores: dict[str, float]
    evidence: dict[str, str]


def select_runs(store: RunStore, plan: SelectionPlan, assess: AssessOOF) -> Selection:
    if not plan.candidates or len(set(plan.candidates)) != len(plan.candidates):
        raise ValueError("選定候補は重複なく 1 件以上必要です")
    if not math.isfinite(plan.tolerance) or plan.tolerance < 0:
        raise ValueError("選定の許容差は有限の非負値が必要です")
    verified = [_verify_cv_run(store, run_id, assess) for run_id in plan.candidates]
    records = [record for record, _ in verified]
    for record in records[1:]:
        compare_runs(records[0], record)
    scores = {record.run_id: score for record, score in verified}
    direction = 1 if records[0].greater_is_better else -1
    best = max(direction * value for value in scores.values())
    winner = next(
        run_id for run_id, value in scores.items() if best - direction * value <= plan.tolerance
    )
    evidence = {record.run_id: record.artifacts[CV_EVIDENCE] for record in records}
    return Selection(plan, winner, scores, evidence)


def verify_selection(
    store: RunStore, plan: SelectionPlan, receipt: Path, assess: AssessOOF
) -> Selection:
    """事前計画から再選定し、保存済み記録の候補・結果・証跡をまとめて照合する。"""
    expected = select_runs(store, plan, assess)
    if _json(json.loads(receipt.read_text(encoding="utf-8"))) != _json(asdict(expected)):
        raise ValueError("選定記録が再計算結果と一致しません")
    return expected


def write_selection(path: Path, selection: Selection) -> None:
    """選定記録を作成する。既存の記録は上書きしない。"""
    with path.open("x", encoding="utf-8") as receipt:
        receipt.write(_json(asdict(selection)) + "\n")


def _verify_export_dependencies(store: RunStore, inputs: Mapping[str, str]) -> None:
    """export Store の配置と cv: 名前空間から、同じコンペの CV Store を特定する。"""
    directory = store.competition_dir
    if (directory.parent.name, directory.name) != ("outputs", "exports"):
        raise ValueError("export Store はコンペの outputs/exports にある必要があります")
    if not inputs or any(not key.startswith("cv:") for key in inputs):
        raise ValueError("export の依存には cv: 名前空間が必要です")
    verify_dependencies(
        RunStore(directory.parent.parent),
        {key.removeprefix("cv:"): digest for key, digest in inputs.items()},
    )


def verify_export(store: RunStore, run_id: str) -> ExperimentRecord:
    """公開済み export Store から、成果物・ログ・CV 依存を再照合する。"""
    record = store.record(run_id)
    evidence, _ = _read_evidence(store, record, EXPORT_EVIDENCE)
    if {item.name for item in evidence.parent.iterdir()} != set(record.artifacts):
        raise ValueError(f"{run_id}: 成果物の一覧が記録と一致しません")
    for name in record.artifacts.keys() - {EXPORT_EVIDENCE}:
        store.artifact_path(run_id, name)
    _verify_export_dependencies(store, record.inputs)
    return record


def publish_export(
    store: RunStore,
    run_id: str,
    *,
    plan: SelectionPlan,
    receipt: Path,
    assess: AssessOOF,
    source: SourceSpec,
    write: Callable[[Path, ExperimentRecord], None],
    required_outputs: tuple[str, ...],
    validate_outputs: Callable[[Path], None],
) -> Path:
    """選定とソースを確認後、選定 run を渡して export し、証跡ごと公開する。"""
    selected = verify_selection(store, plan, receipt, assess)
    candidates = {run_id: store.record(run_id) for run_id in plan.candidates}
    if any(
        record.artifacts.get(CV_EVIDENCE) != selected.evidence[record.run_id]
        for record in candidates.values()
    ):
        raise ValueError("選定後に候補の CV 証跡が変わりました")
    winner = candidates[selected.winner]
    _, expected = _read_evidence(store, winner)
    if source.snapshot() != expected:
        raise ValueError("export のソースまたは設定が選定された CV と一致しません")

    export_store = RunStore(store.competition_dir / "outputs" / "exports")
    inputs = {
        f"cv:{candidate}/{name}": digest
        for candidate, record in candidates.items()
        for name, digest in record.artifacts.items()
    }
    _verify_export_dependencies(export_store, inputs)
    record = replace(
        winner,
        run_id=run_id,
        model=f"export:{winner.model}",
        metric=f"cv_reference:{winner.metric}",
        inputs=inputs,
        hypothesis="選定済みモデルの export",
        changed_one_thing="全学習データでの最終学習または提出生成",
        runtime_sec=None,
        peak_memory_mb=None,
        code_version=code_version(REPO_ROOT),
        late_score=None,
        conclusion="",
        next_experiment="",
    )

    def checked_write(path: Path) -> None:
        write(path, winner)
        if not required_outputs or not set(required_outputs) <= {
            item.name
            for item in path.iterdir()
            if item.is_file() and item.name not in {SELECTION, EXPORT_EVIDENCE}
        }:
            raise ValueError("必須の export 成果物がありません")
        validate_outputs(path)
        if source.snapshot() != expected:
            raise ValueError("export 中にソースまたは設定が変わりました")
        _verify_export_dependencies(export_store, inputs)
        write_selection(path / SELECTION, selected)

    def seal(artifacts: Mapping[str, str]) -> str:
        return (
            _json(
                {
                    "source": expected,
                    "record": _record_identity(record),
                    "artifacts": dict(artifacts),
                }
            )
            + "\n"
        )

    with export_store.reserve(run_id) as reservation:
        return export_store.publish(
            reservation, record, checked_write, seal=(EXPORT_EVIDENCE, seal)
        )
