"""インターネットを無効にした Kaggle 提出用 Notebook を生成する。"""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

SLUG = "cassava-leaf-disease-classification"
DUMMY = """from pathlib import Path

import pandas as pd

sample, = (
    path for path in Path('/kaggle/input').rglob('sample_submission.csv')
    if (path.parent / 'train.csv').is_file()
)
submission = pd.read_csv(sample)
if list(submission.columns) != ['image_id', 'label'] or not submission['image_id'].is_unique:
    raise ValueError('Unexpected submission template')
submission['label'] = 0
submission.to_csv('/kaggle/working/submission.csv', index=False)
"""

INFERENCE = """
def submit(sample_path, weights_dir, output, device='cpu'):
    import hashlib
    import pandas as pd

    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ['image_id', 'label'] or not sample['image_id'].is_unique:
        raise ValueError('Unexpected submission template')
    files = [weights_dir / name for name in EXPECTED]
    for path in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED[path.name]:
            raise ValueError(f'Checkpoint hash mismatch: {path.name}')
    images = [sample_path.parent / 'test_images' / name for name in sample['image_id']]
    probability = ensemble(files, images, SIZE, torch.device(device), FLIP)
    if (probability.shape != (len(images), N_CLASSES)
        or not np.isfinite(probability).all()
        or (probability < 0).any() or (probability > 1).any()
        or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid class probabilities')
    sample['label'] = probability.argmax(axis=1)
    sample.to_csv(output, index=False)
"""


def build(
    owner: str, output: Path, trial: str, source: str, dataset: str | None = None, gpu: bool = False
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "id": "submission",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{owner}/kumite-cassava-{trial.lower()}",
        "title": f"Kumite Cassava {trial}",
        "code_file": "submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,
        "competition_sources": [SLUG],
        "dataset_sources": [dataset] if dataset else [],
    }
    for name, value in (("submit.ipynb", notebook), ("kernel-metadata.json", metadata)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def export(owner: str, output: Path, selection_file: Path, gpu: bool = False) -> None:
    from dataset import DATA
    from train import final, selected_weights

    from kumite.provenance import file_sha256

    selection, checkpoints = selected_weights(selection_file)
    trial = str(selection["trial"])
    run_id = str(selection["run_id"])
    dataset = f"{owner}/kumite-cassava-weights-{run_id.lower()}"
    expected = {path.name: file_sha256(path) for path in checkpoints}
    model_path = Path(__file__).with_name("model.py")
    source = model_path.read_text(encoding="utf-8")
    source += f"\nSIZE = {int(str(selection['size']))}\nEXPECTED = {expected!r}\n"
    source += f"FLIP = {bool(selection.get('flip', False))!r}\n" + INFERENCE
    namespace: dict[str, object] = {}
    exec(compile(source, "cassava-notebook", "exec"), namespace)
    submit = namespace["submit"]
    if not callable(submit):
        raise TypeError("Notebook の提出関数を読み込めません")
    device = "cuda" if gpu else "cpu"
    with TemporaryDirectory() as temporary:
        reference, replay = Path(temporary) / "reference.csv", Path(temporary) / "replay.csv"
        final(selection_file, reference)
        submit(DATA / "sample_submission.csv", checkpoints[0].parent, replay, device)
        if reference.read_bytes() != replay.read_bytes():
            raise ValueError("Notebook とローカル提出の再現結果が一致しません")
    source += f"""
root = Path('/kaggle/input')
sample, = (
    path for path in root.rglob('sample_submission.csv')
    if (path.parent / 'test_images').is_dir()
)
weights, = (path.parent for path in root.rglob('fold-0.pt'))
submit(sample, weights, Path('/kaggle/working/submission.csv'), {device!r})
"""
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        shutil.copyfile(checkpoint, weights_dir / checkpoint.name)
    metadata = {
        "id": dataset,
        "title": f"Kumite Cassava weights {run_id}",
        "licenses": [{"name": "other"}],
    }
    (weights_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    build(owner, output / "kernel", trial, source, dataset, gpu)
    evidence = {
        "run_id": run_id,
        "device": device,
        "selection_sha256": file_sha256(selection_file),
        "model_source_sha256": file_sha256(model_path),
        "notebook_sha256": file_sha256(output / "kernel" / "submit.ipynb"),
        "weights": expected,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.selection is None:
        if args.gpu:
            parser.error("--gpu には --selection が必要です")
        build(args.owner, args.output, "P0", DUMMY)
    else:
        export(args.owner, args.output, args.selection, args.gpu)


if __name__ == "__main__":
    main()
