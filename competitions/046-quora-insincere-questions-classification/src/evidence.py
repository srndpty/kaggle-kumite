"""保存済み成果物と実行条件を照合する。"""

import json
from pathlib import Path

import numpy as np

from kumite import text_features
from kumite.cv_run import Evaluation
from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256
from kumite.runs import RunStore

SOURCE_FILES = {
    "model.py": Path(__file__).with_name("model.py"),
    "text_features.py": Path(text_features.__file__),
    "train.py": Path(__file__).with_name("train.py"),
}
MANIFEST = "cv_manifest.json"


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in SOURCE_FILES.items()}


def manifest(trial_id: str, options: dict[str, bool], sources: dict[str, str]) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "options": options,
        "threshold": 0.5 if trial_id == "E00" else "inner-f1",
        "sources": sources,
    }


def checked_oof(
    store: RunStore, record: ExperimentRecord, evaluation: Evaluation, y: np.ndarray
) -> tuple[np.ndarray, float]:
    if any(
        getattr(record, key) != getattr(evaluation, key)
        for key in (
            "competition_id",
            "metric",
            "greater_is_better",
            "data_hash",
            "split_version",
            "split_hash",
            "seed",
        )
    ):
        raise ValueError(f"{record.run_id}: 評価条件が一致しません")
    oof = store.load_array(record.run_id, "oof.npy")
    if oof.shape != y.shape or not np.isin(oof, [0, 1]).all():
        raise ValueError(f"{record.run_id}: OOF の形状または二値が不正です")
    score = float(evaluation.score(y, oof))
    if not np.isclose(score, record.oof_score, rtol=0, atol=1e-12):
        raise ValueError(f"{record.run_id}: OOF F1 が実験ログと一致しません")
    return oof, score


def check_manifest(
    store: RunStore,
    record: ExperimentRecord,
    trial_id: str,
    options: dict[str, bool],
    sources: dict[str, str],
) -> None:
    saved = json.loads(store.artifact_path(record.run_id, MANIFEST).read_text(encoding="utf-8"))
    if saved != manifest(trial_id, options, sources) or record.params != options:
        raise ValueError(f"{record.run_id}: 試行設定または CV ソースのハッシュが一致しません")


def load_baseline(
    store: RunStore,
    run_id: str,
    evaluation: Evaluation,
    y: np.ndarray,
    sources: dict[str, str],
    tune_sizes: list[int],
) -> dict[str, np.ndarray]:
    record = store.record(run_id)
    check_manifest(store, record, "E00", {}, sources)
    oof, _ = checked_oof(store, record, evaluation, y)
    with np.load(store.artifact_path(run_id, "probabilities.npz")) as saved:
        if (
            str(saved["data_hash"]) != evaluation.data_hash
            or str(saved["split_hash"]) != evaluation.split_hash
        ):
            raise ValueError("E00 の確率のデータまたは分割が一致しません")
        sizes = {"oof": len(y), "coverage": len(y)}
        sizes.update({f"tune_{i}": n for i, n in enumerate(tune_sizes)})
        values = {key: saved[key] for key in sizes}
    for key, n in sizes.items():
        p = values[key]
        if p.shape != (n,) or not np.all((p >= 0) & (p <= 1)):
            raise ValueError(f"E00 の {key} の形状または値が不正です")
    if not np.array_equal(values["oof"] >= 0.5, oof):
        raise ValueError("E00 の閾値 0.5 の再構成が OOF と一致しません")
    return values
