"""公式採点の疎通確認用 Notebook を生成する。"""

import argparse
from pathlib import Path

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "feedback-prize-english-language-learning"
SOURCE = """from pathlib import Path

import pandas as pd

slug = 'feedback-prize-english-language-learning'
root, = (file.parent for file in Path('/kaggle/input').rglob('train.csv')
         if file.parent.name == slug)
train = pd.read_csv(root / 'train.csv', dtype={'text_id': str})
test = pd.read_csv(root / 'test.csv', dtype={'text_id': str})
submission = pd.read_csv(root / 'sample_submission.csv', dtype={'text_id': str})
targets = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
if list(submission.columns) != ['text_id', *targets]:
    raise ValueError('Invalid submission columns')
ids = submission['text_id']
if (ids.empty or ids.isna().any() or not ids.is_unique
        or not test['text_id'].is_unique or set(ids) != set(test['text_id'])):
    raise ValueError('Invalid submission IDs')
means = train[targets].mean()
if not means.between(1, 5).all():
    raise ValueError('Invalid training means')
for target in targets:
    submission[target] = means[target]
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
        kernel_id=f"{args.owner}/kumite-ell-p0",
        title="Kumite ELL P0",
        competition=SLUG,
    )


if __name__ == "__main__":
    main()
