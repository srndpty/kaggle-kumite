"""提出用の非公開 Kaggle Notebook をローカルに生成する。"""

import argparse
import json
import re
import zlib
from base64 import b85encode
from pathlib import Path

from dataset import load_train, train_content_hash
from evidence import checked_folds, notebook_sources, source_hashes
from train import ROOT, TRIALS

from kumite.provenance import array_sha256
from kumite.runs import RunStore

SLUG = "jigsaw-unintended-bias-in-toxicity-classification"
CONSTANT = """from pathlib import Path

import pandas as pd

sample = next(Path("/kaggle/input").rglob("sample_submission.csv"))
pd.read_csv(sample).assign(prediction=0.5).to_csv("/kaggle/working/submission.csv", index=False)
"""

TRAINING = """
import hashlib
import sys
import zlib
from base64 import b85decode
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits


def array_hash(values):
    digest = hashlib.sha256(f"{values.dtype.str}{values.shape}".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def main():
    dataset = import_module("dataset")
    metrics = import_module("metrics")
    model = import_module("model")
    data = next(Path("/kaggle/input").rglob("sample_submission.csv")).parent
    train = dataset.load_train(data)
    if dataset.train_content_hash(train) != TRAIN_CONTENT_HASH:
        raise ValueError("ローカル run と Notebook の学習内容が一致しません")
    test, sample = dataset.load_query(data)
    labels = train[["target", *metrics.IDENTITIES]].to_numpy() >= 0.5
    target = train["target"].to_numpy()
    folds = np.frombuffer(zlib.decompress(b85decode(FOLD_BYTES)), dtype=np.int8)
    if array_hash(folds) != SPLIT_HASH or array_hash(train["id"].to_numpy()) != ID_HASH:
        raise ValueError("ローカル run と Notebook の学習行順または分割が一致しません")
    text = train["comment_text"].to_numpy()
    query = test["comment_text"].to_numpy()
    with threadpool_limits(limits=4):
        prediction = model.predict_ensemble(text, labels, target, folds, query, TRIAL)
    dataset.submission(test, sample, prediction).to_csv(
        "/kaggle/working/submission.csv", index=False
    )


with TemporaryDirectory() as folder:
    root = Path(folder)
    hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in SOURCES.items()
    }
    if hashes != SOURCE_HASHES:
        raise ValueError("Notebook の埋め込みソースが実験時の記録と一致しません")
    for name, content in SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    sys.path.insert(0, str(root))
    main()
"""


def export(owner: str, run_id: str, output: Path) -> Path:
    store = RunStore(ROOT)
    record = store.record(run_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Kaggle のユーザー名が不正です")
    if record.split_version == "none" and record.params.get("prediction") == 0.5:
        source = CONSTANT
    else:
        trial = record.params["trial"]
        if trial not in TRIALS or trial == "E00":
            raise ValueError(f"{run_id} の実験設定を Notebook に変換できません")
        if "notebook_evidence.json" not in record.artifacts:
            raise ValueError(
                f"{run_id} は学習時の Notebook 証跡がありません。新しい run が必要です"
            )
        manifest = json.loads(
            store.artifact_path(run_id, "notebook_evidence.json").read_text(encoding="utf-8")
        )
        sources = notebook_sources()
        if source_hashes(sources) != manifest["source_sha256"]:
            raise ValueError("現在のソースが実験時の記録と一致しません")
        train = load_train(ROOT / "data")
        if train_content_hash(train) != manifest["train_content_sha256"]:
            raise ValueError("現在の学習内容が実験時の記録と一致しません")
        folds = checked_folds(store, record, train["id"])
        fold_bytes = b85encode(zlib.compress(folds.tobytes())).decode("ascii")
        id_hash = array_sha256(train["id"].to_numpy())
        source = (
            f"TRIAL = {trial!r}\nSPLIT_HASH = {record.split_hash!r}\nID_HASH = {id_hash!r}\n"
            f"TRAIN_CONTENT_HASH = {manifest['train_content_sha256']!r}\n"
            f"SOURCE_HASHES = {manifest['source_sha256']!r}\n"
            f"FOLD_BYTES = {fold_bytes!r}\nSOURCES = {sources!r}\n" + TRAINING
        )
    name = f"kumite-042-{run_id.lower().replace('_', '-')}"
    metadata = {
        "id": f"{owner}/{name}",
        "title": f"Kumite 042 {run_id}",
        "code_file": "inference.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [SLUG],
        "kernel_sources": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "inference.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export(args.owner, args.run_id, args.output))
