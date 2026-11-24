"""収録の監査と被験者単位の固定分割を作る。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline import EVENTS, RATES, annotated, read_recording, recordings
from sklearn.model_selection import StratifiedKFold

from kumite.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LABELED_COUNTS = {"tdcsfog": 833, "defog": 91}


def audit(data: Path) -> pd.DataFrame:
    metadata = {
        kind: pd.read_csv(data / f"{kind}_metadata.csv", dtype=str).set_index("Id")
        for kind in RATES
    }
    if any(not frame.index.is_unique for frame in metadata.values()):
        raise ValueError("Duplicate recording ID in metadata")
    paths = recordings(data, "train")
    counts = {kind: sum(path.parent.name == kind for path in paths) for kind in RATES}
    if counts != LABELED_COUNTS:
        raise ValueError(f"Incomplete official training data: {counts} (expected {LABELED_COUNTS})")
    rows: list[dict[str, object]] = []
    checksums: set[str] = set()
    for path in paths:
        kind = path.parent.name
        checksum = file_sha256(path)
        if checksum in checksums:
            raise ValueError(f"Duplicate recording contents: {path.name}")
        checksums.add(checksum)
        subject = metadata[kind].at[path.stem, "Subject"]
        if not isinstance(subject, str) or not subject:
            raise ValueError(f"Missing subject for {path.stem}")
        frame = read_recording(path, labeled=True)
        valid = annotated(frame, kind)
        rows.append(
            {
                "Id": path.stem,
                "kind": kind,
                "subject": subject,
                "rows": len(frame),
                "annotated": int(valid.sum()),
                "gaps": int(frame["Time"].diff().iloc[1:].ne(1).sum()),
                "sha256": checksum,
                **{event: int(frame.loc[valid, event].sum()) for event in EVENTS},
            }
        )
    return pd.DataFrame(rows)


def assign_folds(summary: pd.DataFrame) -> pd.DataFrame:
    subjects = summary.groupby("subject", sort=True)
    cohorts = subjects["kind"].agg(lambda kinds: "+".join(sorted(set(kinds))))
    totals = subjects[EVENTS].sum()
    positives = totals.to_numpy()
    expected = positives.sum(axis=0)
    if (expected == 0).any():
        raise ValueError("Missing event class in training data")
    for count in (5, 4, 3):
        strata = cohorts.where(cohorts.map(cohorts.value_counts()) >= count, "other")
        strata = strata.where(strata.map(strata.value_counts()) >= count, str(strata.mode().iat[0]))
        best: tuple[float, np.ndarray] | None = None
        for seed in range(SEED, SEED + 30):
            folds = np.empty(len(cohorts), dtype=np.int8)
            for fold, (_, valid) in enumerate(
                StratifiedKFold(count, shuffle=True, random_state=seed).split(cohorts, strata)
            ):
                folds[valid] = fold
            counts = np.stack([positives[folds == fold].sum(axis=0) for fold in range(count)])
            if (counts > 0).all():
                imbalance = float(np.square(counts / expected - 1 / count).sum())
                if best is None or imbalance < best[0]:
                    best = imbalance, folds
        if best is not None:
            result = summary.copy()
            result["fold"] = result["subject"].map(pd.Series(best[1], index=cohorts.index))
            return result
    raise ValueError("No subject split with all three events represented in each fold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/split.csv")
    args = parser.parse_args()
    split = assign_folds(audit(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = split.groupby(["kind", "fold"])[["rows", "annotated", "gaps", *EVENTS]].sum()
    print(summary.to_string())
    print(json.dumps({"recordings": len(split), "subjects": split["subject"].nunique()}))
    with args.output.open("x", encoding="utf-8", newline="") as output:
        split.to_csv(output, index=False)


if __name__ == "__main__":
    main()
