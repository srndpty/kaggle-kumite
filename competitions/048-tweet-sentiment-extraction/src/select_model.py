"""保存済み OOF から最終 run の選定と感情・長さ・境界別の集計を行う。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from spans import ID, SEED, TARGET, group_key, load, position
from train import ROOT

from kumite.bootstrap import group_bootstrap_diff
from kumite.provenance import file_sha256
from kumite.runs import RunStore


def select(output: Path, run_suffix: str = "") -> dict[str, object]:
    store = RunStore(ROOT)
    train, _, _ = load(ROOT / "data")
    run_ids = ["E00", "E01", "E02", "E03", "E04", "E05"]
    records = {run_id: store.record(run_id + run_suffix) for run_id in run_ids}
    if len({(record.data_hash, record.split_hash) for record in records.values()}) != 1:
        raise ValueError("比較する run のデータまたは分割が一致しません")
    scores = {run_id: record.oof_score for run_id, record in records.items()}
    # 全文、基本モデル、追加デコード規則、追加特徴の順で同点を解消する。
    preference = ["E00", "E01", "E04", "E05", "E02", "E03"]
    selected = max(preference, key=scores.__getitem__)
    stored = {}
    hashes = {}
    for run_id in run_ids:
        path = store.runs_dir / records[run_id].run_id / "oof.csv"
        digest = file_sha256(path)
        if digest != records[run_id].artifacts["oof.csv"]:
            raise ValueError(f"{run_id} の OOF が実験ログと一致しません")
        frame = pd.read_csv(path, dtype={ID: str}, keep_default_na=False)
        if not frame[ID].equals(train[ID].astype(str)):
            raise ValueError(f"{run_id} の行順が一致しません")
        stored[run_id] = frame
        hashes[run_id] = digest

    best, base = stored[selected], stored["E01"]
    diff = group_bootstrap_diff(
        base["score"].to_numpy(),
        best["score"].to_numpy(),
        train["text"].map(group_key).to_numpy(),
        transform=lambda values: values,
        seed=SEED,
    )

    lengths = train[TARGET].str.split().map(len)
    boundary = []
    for text, answer, prediction in zip(
        train["text"], train[TARGET], best["prediction"], strict=True
    ):
        truth = position(text, answer)
        found = position(text, prediction)
        if answer.strip() == prediction.strip():
            label = "一致"
        elif truth is None or found is None:
            label = "位置が複数"
        elif found[1] <= truth[0] or truth[1] <= found[0]:
            label = "重なりなし"
        elif found[0] <= truth[0] and truth[1] <= found[1]:
            label = "過抽出"
        elif truth[0] <= found[0] and found[1] <= truth[1]:
            label = "不足"
        else:
            label = "一部重複"
        boundary.append(label)

    subsets = {
        "E02": train["text"].str.count(r"([a-zA-Z])\1{2,}").gt(0),
        "E03": train["text"].str.contains(
            r"\b(?:not|never|no|very|really|so|too)\b|n['’]t\b", case=False, regex=True
        ),
    }
    result: dict[str, object] = {
        "hypothesis_subsets": {
            run_id: {
                "n": int(mask.sum()),
                "delta_vs_e01": float((stored[run_id]["score"] - base["score"])[mask].mean()),
            }
            for run_id, mask in subsets.items()
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected": selected,
        "scores": scores,
        "oof_sha256": hashes,
        "delta_vs_e01_ci95": [diff.low, diff.high],
        "by_sentiment": {
            run_id: stored[run_id]["score"].groupby(train["sentiment"]).mean().to_dict()
            for run_id in run_ids
        },
        "by_answer_length": {
            label: {"n": int(mask.sum()), "jaccard": float(best.loc[mask, "score"].mean())}
            for label, mask in {
                "1": lengths == 1,
                "2–3": lengths.between(2, 3),
                "4+": lengths >= 4,
            }.items()
        },
        "by_boundary": best["score"].groupby(boundary).agg(["size", "mean"]).to_dict("index"),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    print(json.dumps(select(args.output, args.run_suffix), ensure_ascii=False, indent=2))
