"""採点疎通用の定数予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "optiver-realized-volatility-prediction"
SOURCE = """from pathlib import Path

import pandas as pd

test_path, = (
    path for path in Path('/kaggle/input').rglob('test.csv')
    if (path.parent / 'book_test.parquet').is_dir()
    and (path.parent / 'trade_test.parquet').is_dir()
)
submission = pd.read_csv(test_path, usecols=['row_id'], dtype={'row_id': str})
row_ids = submission['row_id']
if submission.empty or row_ids.isna().any() or not row_ids.is_unique:
    raise ValueError('Invalid submission IDs')
submission['target'] = 0.003
submission.to_csv('/kaggle/working/submission.csv', index=False)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-optiver-p0",
        title="Kumite Optiver P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
