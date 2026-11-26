"""公式サンプルを使って採点疎通用の定数提出を作る。"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["seg_id", "time_to_failure"]


def write_dummy(data: Path, output: Path) -> None:
    if not (data / "train.csv").is_file():
        raise ValueError("公式 train.csv がありません")
    with (data / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("公式サンプルの列が一致しません")
        ids = [row["seg_id"] for row in reader]
    if not ids or any(not seg_id for seg_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("公式サンプルの ID に空欄または重複があります")
    if {path.stem for path in (data / "test").glob("*.csv")} != set(ids):
        raise ValueError("公式サンプルとテストファイルの ID が一致しません")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows((seg_id, "0.0") for seg_id in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "p0-dummy.csv")
    args = parser.parse_args()
    write_dummy(args.data, args.output)


if __name__ == "__main__":
    main()
