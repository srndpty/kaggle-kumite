"""自己一致だけを返す採点確認用 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "shopee-product-matching"
SOURCE = """import csv
from pathlib import Path

test_file, = (
    path for path in Path('/kaggle/input').rglob('test.csv')
    if path.parent.name == 'shopee-product-matching'
)
with test_file.open(encoding='utf-8', newline='') as source:
    ids = [row['posting_id'] for row in csv.DictReader(source)]
if not ids or any(not item or len(item.split()) != 1 for item in ids) or len(ids) != len(set(ids)):
    raise ValueError('Invalid test posting IDs')
with Path('/kaggle/working/submission.csv').open('w', encoding='utf-8', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['posting_id', 'matches'])
    writer.writerows((item, item) for item in ids)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-shopee-p0",
        title="Kumite Shopee P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
