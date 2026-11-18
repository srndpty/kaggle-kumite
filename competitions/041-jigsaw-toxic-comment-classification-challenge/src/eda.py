"""ラベル分布・重複・文字長を保存する。"""

import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from cv import make_folds, text_key
from dataset import COMPETITION_DIR, DATA_FILES, LABELS, load_data

from kumite.provenance import files_sha256


def label_conflicts(keys, y: np.ndarray) -> dict[str, Any]:
    """同じ文面に異なるラベルが付いたグループと、その全構成行を数える。"""
    frame = pd.DataFrame(y, columns=list(LABELS)).assign(text_key=list(keys))
    grouped = frame.groupby("text_key", sort=False)
    sizes = grouped.size()
    conflicts = grouped[list(LABELS)].nunique().gt(1)
    any_conflict = conflicts.any(axis=1)
    return {
        "duplicate_groups": int(sizes.gt(1).sum()),
        "conflicting_groups": int(any_conflict.sum()),
        "conflicting_rows": int(sizes[any_conflict].sum()),
        "by_label": {
            label: {
                "groups": int(conflicts[label].sum()),
                "rows": int(sizes[conflicts[label]].sum()),
            }
            for label in LABELS
        },
    }


if __name__ == "__main__":
    train, test, _ = load_data()
    keys = train.comment_text.map(text_key)
    y = train[list(LABELS)].to_numpy(dtype=np.int64)
    folds = make_folds(train.comment_text, y)
    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def vocabulary(texts):
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(token_pattern.findall(text.lower()))
        return counts

    train_vocab = vocabulary(train.comment_text)
    test_vocab = vocabulary(test.comment_text)
    test_keys = test.comment_text.map(text_key)
    summary = {
        "data_hash": files_sha256(DATA_FILES),
        "train_rows": len(train),
        "test_rows": len(test),
        "empty_texts": {
            "train": int(train.comment_text.str.strip().eq("").sum()),
            "test": int(test.comment_text.str.strip().eq("").sum()),
        },
        "test_duplicate_rows": int(test_keys.duplicated().sum()),
        "test_length_quantiles": test.comment_text.str.len()
        .quantile([0, 0.5, 0.9, 0.99, 1])
        .to_dict(),
        "vocabulary": {
            "train_types": len(train_vocab),
            "test_types": len(test_vocab),
            "shared_types": len(train_vocab.keys() & test_vocab.keys()),
            "test_unseen_token_share": sum(v for k, v in test_vocab.items() if k not in train_vocab)
            / max(1, sum(test_vocab.values())),
        },
        "positive_rates": train[list(LABELS)].mean().to_dict(),
        "duplicate_rows": int(keys.duplicated().sum()),
        "duplicate_label_conflicts": {
            "exact": label_conflicts(train.comment_text, y),
            "normalized": label_conflicts(keys, y),
        },
        "train_test_shared_texts": len(set(keys) & set(test.comment_text.map(text_key))),
        "length_quantiles": train.comment_text.str.len().quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
        "fold_positive_counts": [y[folds == k].sum(axis=0).tolist() for k in range(3)],
    }
    out = COMPETITION_DIR / "outputs" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    train[list(LABELS)].T.dot(train[list(LABELS)]).to_csv(out / "label_cooccurrence.csv")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
