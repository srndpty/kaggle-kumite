"""manifestの重複・話者・背景録音を診断する。testラベルは読まない。"""

import json

import numpy as np
from dataset import OUTPUT, load_prepared, make_folds


def main() -> None:
    frame, test, _, _, summary = load_prepared()
    shared = set(frame.sha256) & set(test.sha256)
    speech = frame.loc[frame.offset < 0]
    background = frame.loc[frame.offset >= 0]
    durations = speech.samples / speech.rate
    folds = make_folds(frame, 42)
    diagnostics = dict(
        data_hash=summary["data_hash"],
        n_train=len(frame),
        n_test=len(test),
        n_speech=len(speech),
        n_speakers=int(speech.group.nunique()),
        n_background_sources=int(background.group.nunique()),
        n_silence_windows=len(background),
        speech_duration=durations.describe().to_dict(),
        largest_groups=frame.groupby("group").size().nlargest(5).to_dict(),
        duplicate_conflicting_labels=int((speech.groupby("sha256").label.nunique() > 1).sum()),
        shared_hashes=len(shared),
        shared_train_rows=int(frame.sha256.isin(shared).sum()),
        shared_test_rows=int(test.sha256.isin(shared).sum()),
        shared_nonzero_hashes=int(
            frame.loc[frame.sha256.isin(shared) & (frame.rms > 0), "sha256"].nunique()
        ),
        zero_speech_words=speech.loc[speech.rms == 0, "word"].value_counts().to_dict(),
        folds=[
            dict(
                fold=k,
                n_train=int(np.sum(folds != k)),
                n_valid=int(np.sum(folds == k)),
                shared_groups=len(
                    set(frame.loc[folds != k, "group"]) & set(frame.loc[folds == k, "group"])
                ),
            )
            for k in range(3)
        ],
        background=[
            dict(source=group, seconds=len(rows), fold=int(folds[rows.index[0]]))
            for group, rows in background.groupby("group")
        ],
    )
    with (OUTPUT / "eda_review.json").open("x", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
