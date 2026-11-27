"""公式テンプレートの行順で人気商品による P0 の提出を作る。"""

import csv
import json
from collections import Counter
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "p0-dummy.csv"
DUMMY_HISTORY_SESSIONS = 100_000


def main() -> None:
    counts: dict[str, Counter[int]] = {
        action: Counter() for action in ("clicks", "carts", "orders")
    }
    with (DATA / "train.jsonl").open("rb") as source:
        for line in islice(source, DUMMY_HISTORY_SESSIONS):
            for event in json.loads(line)["events"]:
                counts[event["type"]][event["aid"]] += 1
    if not all(counts.values()):
        raise ValueError("学習データに含まれない行動種別があります")
    labels = {action: count.most_common(1)[0][0] for action, count in counts.items()}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    with (
        (DATA / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as source,
        temporary.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.reader(source)
        if next(reader, None) != ["session_type", "labels"]:
            raise ValueError("公式テンプレートの列が session_type,labels と一致しません")
        writer = csv.writer(destination)
        writer.writerow(["session_type", "labels"])
        for key, _ in reader:
            session, action = key.rsplit("_", 1)
            if not session.isdecimal():
                raise ValueError(f"無効なセッション ID: {session}")
            writer.writerow((key, labels[action]))
    temporary.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
