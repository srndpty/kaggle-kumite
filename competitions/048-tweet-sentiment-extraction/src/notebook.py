"""Kaggle に送る、共有実装を含む提出 Notebook を組み立てる。"""

import argparse
import hashlib
import json
from pathlib import Path

from spans import ID, load, make_folds

ROOT = Path(__file__).resolve().parents[1]
SLUG = "tweet-sentiment-extraction"


def build(output: Path, owner: str, trial: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src" / "spans.py").read_text(encoding="utf-8")
    fold_digits = ""
    train_id_hash = ""
    if trial != "E00":
        train, _, _ = load(ROOT / "data")
        fold_digits = "".join(str(fold) for fold in make_folds(train))
        train_id_hash = hashlib.sha256("\n".join(train[ID]).encode()).hexdigest()
    inference = f"""
import hashlib
import platform
import sklearn

print(platform.python_version(), np.__version__, pd.__version__, sklearn.__version__)
data = next(path.parent for path in Path('/kaggle/input').rglob('sample_submission.csv')
            if all((path.parent / name).is_file() for name in FILES))
train, test, sample = load(data)
trial = {trial!r}
if trial == 'E00':
    predictions = test['text'].tolist()
else:
    if hashlib.sha256(chr(10).join(train[ID]).encode()).hexdigest() != {train_id_hash!r}:
        raise ValueError('学習行の ID が固定した分割と一致しません')
    folds = np.fromiter(map(int, {fold_digits!r}), dtype=np.int8)
    print('fold SHA-256:', hashlib.sha256(folds.tobytes()).hexdigest())
    scores = np.mean([score_tokens(test, fit(train.loc[folds != fold], trial), trial)
                      for fold in range(5)], axis=0)
    predictions = decode(test, scores, trial)
write_submission(test, sample, predictions, Path('submission.csv'))
"""
    cells = [
        dict(
            cell_type="code",
            metadata={},
            source=code.splitlines(keepends=True),
            execution_count=None,
            outputs=[],
            id=cell_id,
        )
        for cell_id, code in (("shared", source), ("inference", inference))
    ]
    notebook = dict(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        nbformat=4,
        nbformat_minor=5,
    )
    title = f"Kumite Tweet Sentiment {trial}"
    metadata = {
        "id": f"{owner}/{title.lower().replace(' ', '-')}",
        "title": title,
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [SLUG],
    }
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=[f"E0{i}" for i in range(6)])
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output, args.owner, args.trial)
