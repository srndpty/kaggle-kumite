import csv
import json
from pathlib import Path

import pytest
import torch
from cv_checks import read_sources
from export import write_bundle
from model import SteelUNet, make_presence_model, normalize, training_loss
from PIL import Image
from steel import decode_rle
from train import SteelImages


@pytest.mark.parametrize("variant", ["area", "gate"])
def test_training_step_and_notebook_output(tmp_path: Path, variant: str) -> None:
    torch.set_num_threads(2)
    data = tmp_path / "input"
    test_dir = data / "competition" / "test_images"
    test_dir.mkdir(parents=True)
    image_id = "test.jpg"
    Image.new("RGB", (1600, 256)).save(test_dir / image_id)
    dataset = SteelImages(
        test_dir, {image_id: {1: "1 2"}}, [image_id], (128, 800), resize_masks=True
    )
    pixels, target = dataset[0]
    model = SteelUNet()
    optimizer = torch.optim.AdamW(model.parameters())
    logits = model(normalize(pixels[None], torch.device("cpu")))
    loss = training_loss(logits, target[None].float(), use_dice=True)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    gate = None
    if variant == "gate":
        gate = make_presence_model()
        with torch.no_grad():
            gate.fc.weight.zero_()
            gate.fc.bias.fill_(-1)
    write_bundle(data, model, (128, 800), variant, "owner", "smoke", gate, sources=read_sources())
    notebook = json.loads((data / "submit.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    output = tmp_path / "submission.csv"
    source = source.replace("'/kaggle/input'", json.dumps(str(data)))
    source = source.replace("'/kaggle/working/submission.csv'", json.dumps(str(output)))
    exec(compile(source, "inference-notebook", "exec"), {})
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["ImageId_ClassId"] for row in rows] == [f"{image_id}_{c}" for c in range(1, 5)]
    assert all(
        not row["EncodedPixels"] or decode_rle(row["EncodedPixels"]).sum() >= 32 for row in rows
    )
    if variant == "gate":
        assert all(not row["EncodedPixels"] for row in rows)
