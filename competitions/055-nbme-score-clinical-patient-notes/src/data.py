"""公式 CSV の結合とノート単位の固定分割。"""

import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from spans import Span, normalize

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@dataclass(frozen=True)
class Pair:
    id: str
    case: str
    note_id: str
    feature_id: str
    text: str
    feature: str
    spans: list[Span]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_location(location: str, length: int) -> list[Span]:
    parsed: list[Span] = []
    for entry in ast.literal_eval(location):
        for segment in entry.split(";"):
            start, end = map(int, segment.split())
            parsed.append((start, end))
    return normalize(parsed, length)


def load_pairs(data_dir: Path, *, training: bool) -> list[Pair]:
    notes = {
        (row["case_num"], row["pn_num"]): row["pn_history"]
        for row in read_rows(data_dir / "patient_notes.csv")
    }
    features = {
        (row["case_num"], row["feature_num"]): row["feature_text"]
        for row in read_rows(data_dir / "features.csv")
    }
    pairs: list[Pair] = []
    for row in read_rows(data_dir / ("train.csv" if training else "test.csv")):
        case = row["case_num"]
        text = notes[case, row["pn_num"]]
        pairs.append(
            Pair(
                row["id"],
                case,
                row["pn_num"],
                row["feature_num"],
                text,
                features[case, row["feature_num"]],
                parse_location(row["location"], len(text)) if training else [],
            ),
        )
    if not pairs or len({pair.id for pair in pairs}) != len(pairs):
        msg = "評価 ID が空か重複しています"
        raise ValueError(msg)
    return pairs


def make_folds(pairs: list[Pair]) -> list[int]:
    notes = {pair.note_id: (pair.case, pair.text) for pair in pairs}
    note_ids = sorted(notes)
    cases = [notes[note_id][0] for note_id in note_ids]
    texts = [notes[note_id][1] for note_id in note_ids]
    # 同じ本文の別 ID もまとめ、層化はペア数でなくノート数に対して行う。
    groups = np.unique(texts, return_inverse=True)[1]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignment: dict[str, int] = {}
    for fold, (_, validation) in enumerate(cv.split(note_ids, cases, groups)):
        for index in validation:
            assignment[note_ids[index]] = fold
    return [assignment[pair.note_id] for pair in pairs]


if __name__ == "__main__":
    pairs = load_pairs(ROOT / "data", training=True)
    folds = make_folds(pairs)
    notes = {pair.note_id: pair.text for pair in pairs}
    fold_cases = [
        dict(
            sorted(
                Counter(
                    dict(
                        (pair.note_id, pair.case)
                        for pair, f in zip(pairs, folds, strict=True)
                        if f == fold
                    ).values()
                ).items()
            )
        )
        for fold in range(5)
    ]
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "notes": len(notes),
                "empty": sum(not pair.spans for pair in pairs),
                "multi_span": sum(len(pair.spans) > 1 for pair in pairs),
                "character_quantiles": np.quantile(
                    list(map(len, notes.values())), [0, 0.5, 0.95, 1]
                ).tolist(),
                "duplicates": len(notes) - len(set(notes.values())),
                "fold_cases": fold_cases,
            },
            indent=2,
        )
    )
