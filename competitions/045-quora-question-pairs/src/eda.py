"""公式データと採用した分割の要約。"""

import json

import numpy as np
import pandas as pd
from predict import ROOT, TEXT, read_test
from split import components, make_splits, question_nodes


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    _, test = read_test(ROOT)
    nodes = question_nodes(frame)
    labels = frame["is_duplicate"].to_numpy(dtype=np.int64)
    version, splits = make_splits(nodes, labels, 42)
    _, group_sizes = np.unique(components(nodes), return_counts=True)
    pairs = pd.DataFrame(np.sort(nodes, axis=1), columns=["a", "b"])
    pairs["label"] = labels
    grouped = pairs.groupby(["a", "b"])["label"]
    train_text = pd.Index(pd.concat([frame[column] for column in TEXT]).dropna().unique())
    overlap = test[TEXT[0]].isin(train_text).astype(int) + test[TEXT[1]].isin(train_text).astype(
        int
    )
    result = {
        "train": len(frame),
        "test": len(test),
        "positive_rate": float(labels.mean()),
        "missing": frame[TEXT].isna().sum().to_dict(),
        "shared_nodes": int((np.unique(nodes, return_counts=True)[1] > 1).sum()),
        "duplicate_pair_rows": int(pairs.duplicated(["a", "b"]).sum()),
        "conflicting_pairs": int(grouped.nunique().gt(1).sum()),
        "components": len(group_sizes),
        "largest_component": int(group_sizes.max()),
        "test_seen_endpoints_raw_text": overlap.value_counts().to_dict(),
        "version": version,
        "folds": [
            {
                "train": len(train),
                "valid": len(valid),
                "valid_positive": float(labels[valid].mean()),
            }
            for train, valid in splits
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
