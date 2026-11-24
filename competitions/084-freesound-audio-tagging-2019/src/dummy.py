"""P0 の固定タグ順位を出力する非公開 CPU Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "freesound-audio-tagging-2019"
SOURCE = """from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sample, = Path('/kaggle/input').rglob('sample_submission.csv')
template = pd.read_csv(sample, dtype={'fname': str})
if template.empty or template.columns[0] != 'fname' or len(template.columns) != 81:
    raise ValueError('Unexpected submission template')
if template['fname'].isna().any() or template['fname'].duplicated().any():
    raise ValueError('Invalid test clip IDs')
tags = template.columns[1:]
with ZipFile(sample.with_name('test.zip')) as archive:
    files = sorted(Path(name).name for name in archive.namelist() if name.endswith('.wav'))
if not files:
    raise ValueError('No test clips found')
if set(files) != set(template['fname']):
    template = pd.DataFrame({'fname': files})
scores = np.linspace(0.99, 0.01, len(tags))
template[tags] = np.broadcast_to(scores, (len(template), len(tags)))
template.to_csv('/kaggle/working/submission.csv', index=False)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-freesound-p0",
        title="Kumite Freesound P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
