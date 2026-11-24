"""P0 の定数予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from model import FILES, TARGETS, submission_io_source

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "commonlit-evaluate-student-summaries"
SOURCE = f"""from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {TARGETS!r}
FILES = {FILES!r}

{submission_io_source()}

data, = (
    path.parent for path in Path('/kaggle/input').rglob('summaries_train.csv')
    if all((path.parent / name).is_file() for name in FILES)
)
train, test = load(data)
prediction = np.broadcast_to(train[TARGETS].mean().to_numpy(), (len(test), len(TARGETS)))
write_submission(test, prediction, Path('/kaggle/working/submission.csv'),
                 data / 'sample_submission.csv')
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
