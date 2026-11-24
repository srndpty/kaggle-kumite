"""P0 の定数予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "commonlit-evaluate-student-summaries"
SOURCE = """from pathlib import Path

import pandas as pd

train_path, = (
    path for path in Path('/kaggle/input').rglob('summaries_train.csv')
    if (path.parent / 'summaries_test.csv').is_file()
)
train = pd.read_csv(train_path)
submission = pd.read_csv(train_path.parent / 'summaries_test.csv', dtype={'student_id': str})
submission = submission[['student_id']].copy()
for target in ('content', 'wording'):
    submission[target] = train[target].mean()
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
        kernel_id=f"{args.owner}/kumite-summaries-p0",
        title="Kumite Summaries P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
