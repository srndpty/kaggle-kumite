"""採点確認用の固定予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "child-mind-institute-detect-sleep-states"
SOURCE = """from pathlib import Path

import pandas as pd

test_file, = Path('/kaggle/input').rglob('test_series.parquet')
test = pd.read_parquet(test_file, columns=['series_id', 'step'])
if test.empty or test.isna().any().any():
    raise ValueError('Invalid test series')
starts = test.groupby('series_id', sort=True, as_index=False)['step'].min()
submission = starts.merge(pd.DataFrame({'event': ['onset', 'wakeup']}), how='cross')
submission['score'] = 0.0
submission.insert(0, 'row_id', range(len(submission)))
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
        kernel_id=f"{args.owner}/kumite-sleep-p0",
        title="Kumite Sleep P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
