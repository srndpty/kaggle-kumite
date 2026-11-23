"""公式サンプルと同じ画像 ID 順で P0 の定数提出を作る。"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_submission_v2.csv" / "sample_submission_v2.csv"
OUTPUT = ROOT / "outputs" / "p0-dummy.csv"


def main() -> None:
    with SAMPLE.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image_name", "tags"]:
            raise ValueError("公式サンプルの列が image_name,tags と一致しません")
        ids = [row["image_name"] for row in reader]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("公式サンプルの画像 ID が空、または重複しています")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["image_name", "tags"])
        writer.writerows((image_id, "clear") for image_id in ids)
    print(OUTPUT)


if __name__ == "__main__":
    main()
