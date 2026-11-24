"""実験記録と OOF を照合し、選定結果を固定する。"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from evidence import ROOT, SEED, SPLIT_VERSION, data_digest, inference_sources, source_digests
from inputs import load_data
from model import ALPHA, pearson
from split import check_split, make_split
from train import TRIALS

from kumite.checks import validate_regression
from kumite.provenance import file_sha256
from kumite.runs import RunStore


class Selection(TypedDict):
    selected: str
    run_suffix: str
    data_sha256: str
    split_version: str
    split_sha256: str
    seed: int
    metric: str
    greater_is_better: bool
    source_sha256: dict[str, str]
    oof_sha256: dict[str, str]
    oof_pearson: dict[str, float]


def select(
    root: Path, suffix: str, sources: dict[str, bytes], *, baseline: bool = False
) -> Selection:
    data = root / "data"
    data_hash = data_digest(data)
    train, _ = load_data(data)
    split_path = root / "outputs" / f"{SPLIT_VERSION}.csv"
    split_hash = file_sha256(split_path)
    check_split(split_path, make_split(train))
    y = train["score"].to_numpy(dtype=np.float64)
    hashes = source_digests(sources)
    store = RunStore(root)
    scores: dict[str, float] = {}
    oof_hashes: dict[str, str] = {}
    for trial in ("E01",) if baseline else TRIALS:
        run_id = f"{trial}{suffix}"
        record = store.record(run_id)
        if (
            record.competition_id,
            record.metric,
            record.greater_is_better,
            record.split_version,
            record.seed,
            record.model,
            record.data_version,
        ) != ("054", "pearson", True, SPLIT_VERSION, SEED, "Ridge", "official"):
            raise ValueError(f"{run_id} の実験設定が一致しません")
        if record.data_hash != data_hash or record.split_hash != split_hash:
            raise ValueError(f"{run_id} のデータまたは分割のハッシュが一致しません")
        if record.params.get("source_sha256") != hashes:
            raise ValueError(
                f"{run_id} の推論ソース SHA256 が一致しません。新しい run ID で再評価してください"
            )
        if record.params.get("alpha") != ALPHA or record.params.get("features") != list(
            TRIALS[trial][0]
        ):
            raise ValueError(f"{run_id} のモデル構成が一致しません")
        prediction = store.load_array(run_id, "oof.npy")
        validate_regression(prediction, len(y), lower=0)
        if (prediction > 1).any():
            raise ValueError(f"{run_id} の OOF が範囲外です")
        score = pearson(y, prediction)
        if not np.isclose(score, record.oof_score, rtol=0, atol=1e-9):
            raise ValueError(f"{run_id} の OOF Pearson が実験ログと一致しません")
        scores[run_id] = record.oof_score
        oof_hashes[run_id] = record.artifacts["oof.npy"]
    if data_digest(data) != data_hash or file_sha256(split_path) != split_hash:
        raise ValueError("選定中にデータまたは分割が変わりました")
    return {
        "selected": max(scores, key=lambda run: scores[run]),
        "run_suffix": suffix,
        "data_sha256": data_hash,
        "split_version": SPLIT_VERSION,
        "split_sha256": split_hash,
        "seed": SEED,
        "metric": "pearson",
        "greater_is_better": True,
        "source_sha256": hashes,
        "oof_sha256": oof_hashes,
        "oof_pearson": scores,
    }


def freeze_selection(path: Path, selection: Selection) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--output", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    selection = select(ROOT, args.run_suffix, inference_sources(ROOT))
    freeze_selection(args.output, selection)
    print(selection["selected"])


if __name__ == "__main__":
    main()
