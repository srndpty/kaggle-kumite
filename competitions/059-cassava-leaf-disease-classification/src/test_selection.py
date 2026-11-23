"""選定記録と公式入力が変わった場合の提出拒否を検査する。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import train
from submission_input import submission_inputs

from kumite.provenance import file_sha256


def test_selection_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    original = b'{"run_id": "E04"}\n'
    path.write_bytes(original)
    digest = file_sha256(path)
    with pytest.raises(FileExistsError):
        train.select(["E01"], path)
    assert path.read_bytes() == original
    assert file_sha256(path) == digest


@pytest.mark.parametrize("change", ["weights", "source", "image", "label"])
def test_final_rejects_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    monkeypatch.setattr(train, "DATA", tmp_path)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image_id,label\n1.jpg,0\n", encoding="utf-8")
    images = tmp_path / "train_images"
    images.mkdir()
    image = images / "1.jpg"
    image.write_bytes(b"training image")
    weights = {"E04/fold-0.pt": "original"}
    record = SimpleNamespace(
        params={
            "trial": "E04",
            "size": 320,
            "flip": False,
            "model_source_sha256": file_sha256(Path(train.__file__).with_name("model.py")),
        },
        split_hash="split",
        data_hash=train.training_hash(pd.read_csv(train_csv)),
    )
    monkeypatch.setattr(
        train, "STORE", SimpleNamespace(record=lambda _: record, input_hashes=lambda *_: weights)
    )
    selection = {
        "run_id": "E04",
        "trial": "E04",
        "size": 320,
        "inference_horizontal_flip": False,
        "split_hash": "split",
        "submission_input": {},
        "weights": weights.copy(),
    }
    errors = {
        "weights": "重み",
        "source": "学習設定",
        "image": "公式学習入力",
        "label": "公式学習入力",
    }
    if change == "weights":
        selection["weights"] = {"E04/fold-0.pt": "changed"}
    elif change == "source":
        source = tmp_path / "model.py"
        source.write_text("# changed\n", encoding="utf-8")
        monkeypatch.setattr(train, "__file__", str(tmp_path / "train.py"))
    elif change == "image":
        image.write_bytes(b"changed image")
    else:
        train_csv.write_text("image_id,label\n1.jpg,1\n", encoding="utf-8")
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match=errors[change]):
        train.final(path, output)
    assert not output.exists()


def test_submission_manifest_and_hidden_input(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    images = tmp_path / "test_images"
    images.mkdir()
    public = images / "1.jpg"
    public.write_bytes(b"public image")
    sample.write_text("image_id,label\n1.jpg,0\n", encoding="utf-8")
    _, _, manifest = submission_inputs(sample)
    public.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="証跡"):
        submission_inputs(sample, manifest, allow_hidden=True)
    public.rename(images / "2.jpg")
    sample.write_text("image_id,label\n2.jpg,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="証跡"):
        submission_inputs(sample, manifest)
    submission_inputs(sample, manifest, allow_hidden=True)
    (images / "3.jpg").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="ファイル集合"):
        submission_inputs(sample, manifest, allow_hidden=True)
