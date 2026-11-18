"""同じ行・分割での比較と、クラス・話者・音声条件別のエラー分析。"""

import argparse
import json

import numpy as np
import pandas as pd
from dataset import OUTPUT, ROOT, load_prepared
from train import metrics

from kumite.experiments import compare_runs
from kumite.runs import RunStore


def analyze(base: str, candidate: str) -> dict:
    store = RunStore(ROOT)
    comparison = compare_runs(store.record(base), store.record(candidate))
    frame, _, waves, _, summary = load_prepared()
    if store.record(base).data_hash != summary["data_hash"]:
        raise ValueError("分析対象のデータが一致しません")
    y = frame.label.to_numpy(dtype=np.int64)
    first, second = (store.load_array(run, "oof.npy") for run in (base, candidate))
    result: dict = dict(
        base=base,
        candidate=candidate,
        fold_differences=comparison.diffs,
        metrics={base: metrics(y, first), candidate: metrics(y, second)},
    )
    rms = np.concatenate(
        [
            np.sqrt(np.mean(np.asarray(waves[start : start + 512]) ** 2, axis=1))
            for start in range(0, len(waves), 512)
        ]
    )
    duration = np.where(frame.offset.to_numpy() >= 0, 1.0, frame.samples / frame.rate)
    groups = dict(
        word=frame.word,
        speaker=frame.group,
        duration=pd.Series(pd.cut(duration, [0, 0.75, 0.99, 1.01, np.inf]).astype(str)),
        loudness=pd.Series(pd.qcut(rms, 4, duplicates="drop").astype(str)),
    )
    result["groups"] = {}
    for name, series in groups.items():
        values = {}
        for group in sorted(series.unique()):
            mask = (series == group).to_numpy()
            values[group] = dict(
                n=int(mask.sum()),
                base_accuracy=float(np.mean(first[mask].argmax(1) == y[mask])),
                candidate_accuracy=float(np.mean(second[mask].argmax(1) == y[mask])),
            )
        result["groups"][name] = values
    loss = -np.log(np.clip(second[np.arange(len(y)), y], 1e-15, 1))
    result["high_loss"] = frame.iloc[np.argsort(loss)[-30:][::-1]][["id", "word", "group"]].to_dict(
        "records"
    )
    out = OUTPUT / "analysis"
    out.mkdir(exist_ok=True)
    with (out / f"{base}-{candidate}.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {key: result[key] for key in ("base", "candidate", "fold_differences", "metrics")},
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("candidate")
    args = parser.parse_args()
    analyze(args.base, args.candidate)
