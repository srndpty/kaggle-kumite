"""公式サンプルと同じ ID・列順で P0 の定数予測を作る。"""

import csv
from pathlib import Path

COMPETITION = Path(__file__).resolve().parents[1]


def main() -> None:
    sample = COMPETITION / "data" / "sample_submission.csv"
    output = COMPETITION / "outputs" / "p0.csv"
    output.parent.mkdir(exist_ok=True)
    with sample.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if header != ["row_id", "meter_reading"]:
            raise ValueError(f"公式サンプルの列が想定外です: {header}")
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(header)
            for row_id, _ in reader:
                writer.writerow((row_id, 0))
        temporary.replace(output)


if __name__ == "__main__":
    main()
