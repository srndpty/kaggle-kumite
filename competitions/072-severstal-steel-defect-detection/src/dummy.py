"""全空マスクを出す P0 用の Kaggle Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SOURCE = """import csv
from pathlib import Path

test_dir, = (path for path in Path('/kaggle/input').rglob('test_images') if path.is_dir())
image_ids = sorted(path.name for path in test_dir.glob('*.jpg'))
if not image_ids:
    raise ValueError('No test JPEGs found')
with Path('/kaggle/working/submission.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(('ImageId_ClassId', 'EncodedPixels'))
    for image_id in image_ids:
        for class_id in range(1, 5):
            writer.writerow((f'{image_id}_{class_id}', ''))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_cpu_notebook(
        args.output,
        SOURCE,
        kernel_id=f"{args.owner}/kumite-severstal-p0",
        title="Kumite Severstal P0",
        competition="severstal-steel-defect-detection",
    )


if __name__ == "__main__":
    main()
