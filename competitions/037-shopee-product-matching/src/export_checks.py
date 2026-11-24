"""選定記録、推論ソース、提出バンドルの検証。"""

import hashlib
import json
from pathlib import Path

from kumite.experiments import ExperimentRecord
from kumite.provenance import file_sha256
from kumite.runs import RunStore

SOURCE_FILES = ("matching.py", "images.py", "inference.py")
CRITERION = "外側 OOF の平均 F1 最大。同点の場合は Baseline からの変更が少ない方を優先。"
CHANGE_COUNTS = {"E00": 0, "E01": 1, "E02": 1, "E03": 2, "E04": 3}


def source_hashes(directory: Path) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in SOURCE_FILES}


def verify_sources(record: ExperimentRecord, hashes: dict[str, str]) -> None:
    if any(record.inputs.get(f"source/{name}") != digest for name, digest in hashes.items()):
        raise ValueError(
            "CV の推論ソース SHA256 が未記録または不一致です。新しい run ID で評価してください"
        )


def verify_selection(store: RunStore, run_id: str, path: Path) -> None:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["selected_run"] != run_id:
        raise ValueError("最終 export には選定済み run を指定してください")
    scores = selection["oof_scores"]
    if not scores or run_id not in scores or selection["criterion"] != CRITERION:
        raise ValueError("選定候補または採用規則が不正です")
    records = {key: store.record(key) for key in scores}
    selected = records[run_id]
    for key, record in records.items():
        if (
            record.oof_score != scores[key]
            or record.metric != selection["metric"]
            or record.metric != "mean_f1"
            or not record.greater_is_better
            or record.competition_id != selected.competition_id
            or record.data_hash != selected.data_hash
            or record.split_hash != selected.split_hash
            or record.split_version != selection["split_version"]
        ):
            raise ValueError(f"選定記録と RunStore の評価条件または OOF が不一致です: {key}")
    if {r.params["trial"] for r in records.values()} != set(CHANGE_COUNTS):
        raise ValueError("選定には E00〜E04 の比較結果が必要です")
    best = min(
        records.values(), key=lambda r: (-r.oof_score, CHANGE_COUNTS[r.params["trial"]], r.run_id)
    )
    if best.run_id != run_id:
        raise ValueError("選定済み run が採用規則に一致しません")


def verify_bundle(bundle: Path, expected_manifest_sha256: str) -> None:
    # Notebook にも同じ関数を埋め込むため、標準ライブラリだけで検証する。
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("提出マニフェストの SHA256 が不一致です")
    for name, expected in json.loads(manifest_bytes).items():
        with (bundle / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"提出ファイルの SHA256 が不一致です: {name}")
