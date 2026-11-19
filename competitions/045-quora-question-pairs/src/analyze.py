"""保存済み OOF の単語類似度帯・ラベル別の損失を集計する。"""

import argparse
import json

import numpy as np
from features import NEGATION, NUMBER, TOKEN, PairFeatures
from inputs import read_verified_train
from predict import ROOT
from train import pair_text, score

from kumite.runs import RunStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    store = RunStore(ROOT)
    frame = read_verified_train(ROOT, store.record(args.run_id).data_hash)
    labels = frame["is_duplicate"].to_numpy(dtype=np.int64)
    pairs = pair_text(frame)
    oof = store.load_array(args.run_id, "oof.npy")
    # E02 は文字類似度を追加するため、単語類似度のみを再算出すれば足りる。
    similarities = np.full(len(frame), np.nan)
    with np.load(store.artifact_path(args.run_id, "split.npz")) as split:
        if not np.array_equal(split["ids"], frame["id"].to_numpy()):
            raise ValueError("学習データの行 ID が保存済み split と一致しません")
        for roles in split["roles"]:
            features = PairFeatures("E01")
            features.fit(pairs[roles == 1])
            similarities[roles == 2] = features.transform(pairs[roles == 2])[:, 0]
    if not np.array_equal(np.isfinite(oof), np.isfinite(similarities)):
        raise ValueError("OOF と検証対象が一致しません")
    bands = np.digitize(similarities, [0.25, 0.5, 0.75])
    result = []
    for band in range(4):
        for label in (0, 1):
            mask = np.isfinite(oof) & (bands == band) & (labels == label)
            if mask.any():
                result.append(
                    {
                        "band": band,
                        "label": label,
                        "rows": int(mask.sum()),
                        "log_loss": score(labels[mask], oof[mask]),
                    }
                )
    valid = np.isfinite(oof)
    containment = []
    mismatch = []
    for left, right in pairs:
        a, b = set(TOKEN.findall(left)), set(TOKEN.findall(right))
        containment.append(bool(a and b and (a < b or b < a)))
        mismatch.append(
            set(NUMBER.findall(left)) != set(NUMBER.findall(right))
            or bool(NEGATION.search(left)) != bool(NEGATION.search(right))
        )
    segments = {
        "positive_low_cosine": valid & (labels == 1) & (similarities < 0.25),
        "negative_containment": valid & (labels == 0) & np.array(containment),
        "negative_high_cosine_mismatch": valid
        & (labels == 0)
        & (similarities >= 0.75)
        & np.array(mismatch),
    }
    print(
        json.dumps(
            {
                "similarity_bands": result,
                "segments": {
                    name: {"rows": int(mask.sum()), "log_loss": score(labels[mask], oof[mask])}
                    for name, mask in segments.items()
                    if mask.any()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
