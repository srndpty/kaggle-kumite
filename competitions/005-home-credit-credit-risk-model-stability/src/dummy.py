"""公式テストの ID を使う定数予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "home-credit-credit-risk-model-stability"
SOURCE = """from pathlib import Path

import pandas as pd

test, = Path('/kaggle/input').rglob('test_base.csv')
submission = pd.read_csv(test, usecols=['case_id'], dtype={'case_id': 'int64'})
if submission.empty or not submission['case_id'].is_unique:
    raise ValueError('公式テストの case_id が空または重複しています')
submission['score'] = 0.5
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
        kernel_id=f"{args.owner}/kumite-home-credit-stability-p0",
        title="Kumite Home Credit Stability P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
