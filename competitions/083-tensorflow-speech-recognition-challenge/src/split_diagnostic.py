"""同じ行集合で、分割方式によるスコア差と話者・背景録音共有を記録する。"""

import json

import numpy as np
import pandas as pd
from dataset import CLASSES, OUTPUT, ROOT, load_prepared, make_folds

from kumite.runs import RunStore


def distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1])
    return dict(
        count=len(values),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        **dict(zip(("min", "p25", "median", "p75", "p95", "max"), quantiles.tolist(), strict=True)),
    )


def fold_profiles(frame: pd.DataFrame, folds: np.ndarray, rms: np.ndarray) -> list[dict]:
    """RMSは正規化した1秒窓ごとの値。元の長い背景録音のRMSを流用しない。"""
    result = []
    for k in range(3):
        roles = {}
        for role, mask in (("train", folds != k), ("valid", folds == k)):
            rows = frame.loc[mask]
            speech = rows.loc[rows.label != 10]
            noise = rows.loc[rows.label == 10]
            sources = []
            for path, recording in noise.groupby("path"):
                sources.append(
                    dict(
                        path=path,
                        windows=len(recording),
                        original_seconds=float(recording.samples.iloc[0] / recording.rate.iloc[0]),
                        window_rms=distribution(rms[recording.index.to_numpy()]),
                    )
                )
            speakers = {str(path).split("/")[-1].split("_nohash_")[0] for path in speech.path}
            roles[role] = dict(
                n_rows=len(rows),
                n_speakers=len(speakers),
                n_speaker_groups=int(speech.group.nunique()),
                class_counts={name: int(np.sum(rows.label == i)) for i, name in enumerate(CLASSES)},
                silence_windows=len(noise),
                background_recordings=sources,
                speech_duration_seconds=distribution((speech.samples / speech.rate).to_numpy()),
                speech_rms=distribution(rms[speech.index.to_numpy()]),
                silence_rms=distribution(rms[noise.index.to_numpy()]),
            )
        result.append(dict(fold=k, **roles))
    return result


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
