"""公式 ZIP を照合し、波形を CSV の ID 順の NumPy 配列にまとめる。"""

import io
import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np
from dataset import SHAPE, read_ids, validate_waveform, waveform_path

ROOT = Path(__file__).resolve().parents[1]


def prepare(archive: Path, output: Path) -> None:
    output = output.parent.resolve() / output.name
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(f".{output.name}-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for name in ("training_labels.csv", "sample_submission.csv"):
                (stage / name).write_bytes(bundle.read(name))
            train, _ = read_ids(stage / "training_labels.csv", training=True)
            test, _ = read_ids(stage / "sample_submission.csv")
            if set(train).intersection(test):
                raise ValueError("学習とテストの ID が重複しています")
            if len(members) != len(train) + len(test) + 2:
                raise ValueError("ZIP と CSV の件数が一致しません")
            for split, ids in (("test", test), ("train", train)):
                rows = {
                    waveform_path(Path(), split, key).as_posix(): index
                    for index, key in enumerate(ids)
                }
                out = np.lib.format.open_memmap(
                    stage / f"{split}.npy", mode="w+", dtype="<f8", shape=(len(ids), *SHAPE)
                )
                try:
                    count = 0
                    for member in members:
                        if not member.filename.startswith(f"{split}/"):
                            continue
                        index = rows.pop(member.filename)
                        wave = np.load(io.BytesIO(bundle.read(member)), allow_pickle=False)
                        validate_waveform(wave, member.filename)
                        out[index] = wave
                        count += 1
                        if count % 50000 == 0:
                            print(f"{split}: {count}/{len(ids)} 件", flush=True)
                    if rows:
                        raise ValueError(f"{split} の波形が ZIP にありません")
                    out.flush()
                finally:
                    del out
        stage.rename(output)
    except BaseException:
        if stage.resolve().parent == output.parent:
            shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    prepare(ROOT / "data" / "official.zip", ROOT / "data" / "packed")
