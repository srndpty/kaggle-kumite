"""P0 の固定矩形を出力する Kaggle Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "global-wheat-detection"
SOURCE = """from pathlib import Path

import pandas as pd
from PIL import Image

sample, = (
    path for path in Path('/kaggle/input').rglob('sample_submission.csv')
    if (path.parent / 'test').is_dir()
)
template = pd.read_csv(sample, dtype={'image_id': str})
if list(template.columns) != ['image_id', 'PredictionString']:
    raise ValueError('Unexpected submission columns')
images = sorted((sample.parent / 'test').glob('*.jpg'))
if not images:
    raise ValueError('No test JPEGs found')
rows = []
for path in images:
    with Image.open(path) as image:
        width, height = image.size
    box_width, box_height = min(100, width), min(100, height)
    x, y = (width - box_width) // 2, (height - box_height) // 2
    rows.append((path.stem, f'0.5 {x} {y} {box_width} {box_height}'))
pd.DataFrame(rows, columns=template.columns).to_csv(
    '/kaggle/working/submission.csv', index=False
)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-wheat-p0",
        title="Kumite Wheat P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
