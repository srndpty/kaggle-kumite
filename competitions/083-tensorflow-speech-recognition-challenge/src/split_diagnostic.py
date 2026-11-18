"""同じ行集合で、分割方式によるスコア差と話者・背景録音共有を記録する。"""

import json

import numpy as np
import pandas as pd
from dataset import OUTPUT, ROOT, load_prepared, make_folds

from kumite.runs import RunStore


def main() -> None:
    frame, _, _, _, summary = load_prepared()
    store = RunStore(ROOT)
    result = {}
    for run_id, random_split in (("E01", False), ("E01-random", True)):
        record = store.record(run_id)
        if record.data_hash != summary["data_hash"]:
            raise ValueError("データが一致しません")
        folds = make_folds(frame, 42, random_split)
        detail = []
        for k in range(3):
            train, valid = frame.loc[folds != k], frame.loc[folds == k]
            speech = valid.loc[valid.label != 10]
            noise = valid.loc[valid.label == 10]
            shared = set(train.group) & set(valid.group)
            detail.append(
                dict(
                    fold=k,
                    shared_groups=len(shared),
                    speech_rows_with_seen_speaker=float(speech.group.isin(shared).mean()),
                    seen_speaker_fraction=len(set(speech.group) & shared) / speech.group.nunique(),
                    silence_rows_with_seen_recording=float(noise.group.isin(shared).mean()),
                )
            )
        metrics = json.loads(
            store.artifact_path(run_id, "metrics.json").read_text(encoding="utf-8")
        )
        result[run_id] = dict(
            fold_mean_accuracy=float(np.mean(record.fold_scores)),
            fold_scores=record.fold_scores,
            oof_accuracy=record.oof_score,
            metrics=metrics,
            overlap=detail,
        )
    out = OUTPUT / "split_diagnostic.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        pd.DataFrame(
            {
                key: {
                    "Accuracy": value["fold_mean_accuracy"],
                    "LogLoss": value["metrics"]["logloss"],
                }
                for key, value in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
