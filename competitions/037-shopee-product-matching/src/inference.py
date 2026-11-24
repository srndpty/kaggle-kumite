"""保存した検索器で Kaggle のテスト出品を照合する。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster
from matching import MAX_MATCHES, accepted, mutual_edges, rerank, top_candidates
from sklearn.feature_extraction.text import TfidfVectorizer


def submission(data: Path, bundle: Path) -> pd.DataFrame:
    gallery = pd.read_csv(data / "test.csv")
    if gallery.empty or gallery["posting_id"].isna().any() or not gallery["posting_id"].is_unique:
        raise ValueError("テスト出品 ID は欠損なしで一意である必要があります")
    input_order = gallery["posting_id"].tolist()
    gallery = gallery.sort_values("posting_id")
    ids = gallery["posting_id"].to_numpy()
    if any(not item or len(item.split()) != 1 for item in ids):
        raise ValueError("テスト出品 ID に空白があります")
    config = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    encoders = []
    for state in config["text"]:
        encoder = TfidfVectorizer(
            analyzer=state["analyzer"],
            ngram_range=tuple(state["ngram_range"]),
            vocabulary=state["vocabulary"],
        )
        encoder.fit(["dummy"])
        encoder.idf_ = np.asarray(state["idf"])
        encoders.append(encoder)
    images = None
    weights = bundle / "resnet18-f37072fd.pth"
    if config["weights_sha256"] is not None:
        from images import embed

        with weights.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != config["weights_sha256"]:
                raise ValueError("画像モデルの重みが学習時と一致しません")
        names = gallery["image"].unique().tolist()
        values = embed(data / "test_images", names, weights)
        positions = {name: index for index, name in enumerate(names)}
        images = values[[positions[name] for name in gallery["image"]]]
    candidates = top_candidates(
        tuple(encoders),
        gallery["title"].fillna("").tolist(),
        images=images,
        hashes=gallery["image_phash"].to_numpy(),
    )
    if config["model"] is not None:
        candidates = rerank(candidates, Booster(model_str=config["model"]))
    keep = accepted(candidates, config["threshold"])
    if config["trial"] == "E04":
        source, slot = np.nonzero(keep)
        keep[source, slot] = mutual_edges(source, candidates.indices[source, slot])
    predicted = [
        [item, *ids[indices[selected]].tolist()]
        for item, indices, selected in zip(ids, candidates.indices, keep, strict=True)
    ]
    known = set(ids)
    for item, matches in zip(ids, predicted, strict=True):
        unique = set(matches)
        if item not in unique or len(matches) != len(unique) or len(matches) > MAX_MATCHES:
            raise ValueError("自己一致、重複、または提出件数の検査に失敗しました")
        if not unique <= known:
            raise ValueError("未知の出品 ID が含まれています")
    return (
        pd.DataFrame({"posting_id": ids, "matches": [" ".join(matches) for matches in predicted]})
        .set_index("posting_id")
        .loc[input_order]
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    submission(args.data, args.bundle).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
