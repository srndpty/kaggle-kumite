"""選定・外部評価・Notebook で使う実験の照合。"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from dataset import SEED, SPLIT_VERSION, validate
from metrics import score

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256
from kumite.runs import RunStore

METRIC = "Jigsaw bias AUC"


def notebook_sources() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        **{
            name: (directory / name).read_text(encoding="utf-8")
            for name in ("dataset.py", "metrics.py", "model.py")
        },
        "kumite/__init__.py": "",
        "kumite/text_features.py": (REPO_ROOT / "src" / "kumite" / "text_features.py").read_text(
            encoding="utf-8"
        ),
    }


def source_hashes(sources: dict[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(source.encode("utf-8")).hexdigest() for name, source in sources.items()
    }


def checked_folds(store: RunStore, record: ExperimentRecord, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(store.artifact_path(record.run_id, "folds.csv"), usecols=["id", "fold"])
    if (
        not frame["id"].equals(ids)
        or set(frame["fold"].unique()) != {0, 1, 2}
        or not pd.api.types.is_integer_dtype(frame["fold"])
    ):
        raise ValueError(f"{record.run_id} の学習行順または fold が不正です")
    folds = frame["fold"].to_numpy(dtype=np.int8)
    if array_sha256(folds) != record.split_hash:
        raise ValueError(f"{record.run_id} の fold ハッシュが一致しません")
    return folds


def checked_oof(
    store: RunStore,
    record: ExperimentRecord,
    data_hash: str,
    split_hash: str,
    ids: pd.Series,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    expected = record.run_id.split("-", 1)[0]
    if (
        expected not in {"E01", "E02", "E03", "E04", "E05"}
        or record.params.get("trial") != expected
    ):
        raise ValueError(f"{record.run_id} の trial が一致しません")
    if (record.competition_id, record.metric, record.greater_is_better, record.seed) != (
        "042",
        METRIC,
        True,
        SEED,
    ):
        raise ValueError(f"{record.run_id} の評価契約が一致しません")
    if (record.data_hash, record.split_version, record.split_hash) != (
        data_hash,
        SPLIT_VERSION,
        split_hash,
    ):
        raise ValueError(f"{record.run_id} のデータまたは分割が一致しません")
    checked_folds(store, record, ids)
    prediction = store.load_array(record.run_id, "oof.npy")
    validate(prediction, len(ids))
    value = score(labels, prediction)
    if not math.isclose(value, record.oof_score, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{record.run_id} の OOF スコアが実験ログと一致しません")
    return prediction, value
