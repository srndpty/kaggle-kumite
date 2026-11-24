"""採点確認用の定数予測 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "tlvmc-parkinsons-freezing-gait-prediction"
SOURCE = """from pathlib import Path

import pandas as pd

test_dir, = (path for path in Path('/kaggle/input').rglob('test')
             if (path / 'tdcsfog').is_dir() or (path / 'defog').is_dir())
ids = []
for path in sorted(test_dir.rglob('*.csv'), key=lambda path: path.stem):
    time = pd.read_csv(path, usecols=['Time'], dtype={'Time': 'int64'})['Time']
    ids.append(path.stem + '_' + time.astype(str))
submission = pd.DataFrame({'Id': pd.concat(ids, ignore_index=True)})
if submission.empty or submission['Id'].isna().any() or not submission['Id'].is_unique:
    raise ValueError('Invalid submission IDs')
submission[['StartHesitation', 'Turn', 'Walking']] = 0
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
        kernel_id=f"{args.owner}/kumite-fog-p0",
        title="Kumite FOG P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
