"""採点疎通用の CPU Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "google-quest-challenge"
SOURCE = """import csv
from hashlib import sha256
from pathlib import Path


def make_submission(sample, test, output):
    with test.open(encoding="utf-8", newline="") as stream:
        ids = [row["qa_id"] for row in csv.DictReader(stream)]
    with sample.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        sample_ids = [row[0] for row in reader]
    if len(header) != 31 or header[0] != "qa_id" or len(set(header)) != 31:
        raise ValueError("Invalid sample columns")
    if not ids or "" in ids or len(ids) != len(set(ids)) or set(ids) != set(sample_ids):
        raise ValueError("Invalid test IDs")
    if len(sample_ids) != len(ids):
        raise ValueError("Duplicate submission IDs")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for qa_id in sample_ids:
            values = [
                int.from_bytes(sha256(f"{qa_id}:{column}".encode()).digest()[:4], "big")
                / (2**32 - 1)
                for column in header[1:]
            ]
            writer.writerow([qa_id, *values])


if __name__ == "__main__":
    sample, = (
        path for path in Path("/kaggle/input").rglob("sample_submission.csv")
        if path.parent.name == "google-quest-challenge"
    )
    make_submission(sample, sample.with_name("test.csv"), Path("/kaggle/working/submission.csv"))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-quest-p0",
        title="Kumite QUEST P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
