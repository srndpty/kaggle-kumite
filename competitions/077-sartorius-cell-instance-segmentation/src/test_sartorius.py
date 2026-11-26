import csv
import json
from pathlib import Path
from typing import cast

import model
import numpy as np
import pytest
import torch
import train
from data import Image, dataset
from export import NOTEBOOK, WEIGHTS, selected_trial, validate, write_notebook
from masks import disjoint, encode, image_score, ious, pixels
from model import build, settings
from PIL import Image as PillowImage
from torchvision.models.detection import MaskRCNN
from train import read_rows

from kumite.quality_gates import SourceSpec, publish_export, select_runs, write_selection
from kumite.runs import RunStore


def test_row_major_rle() -> None:
    mask = np.array([[False, True, True], [True, False, True]])
    encoded = encode(np.flatnonzero(mask))
    assert encoded == "2 3 6 1"
    assert pixels(encoded, mask.size).tolist() == [1, 2, 3, 5]
    with pytest.raises(ValueError, match="範囲"):
        pixels("6 2", mask.size)


def test_score_matches_strictly_and_once() -> None:
    assert image_score(ious(["1 2"], ["1 2"], 4)) == 1
    assert image_score(ious(["1 2"], [], 4)) == 0
    assert image_score(ious(["1 2"], ["1 1"], 4)) == 0
    assert image_score(ious(["1 2", "1 2"], ["1 2"], 4)) == 0.5
    with pytest.raises(ValueError, match="重複"):
        ious(["1 2"], ["1 2", "2 2"], 4)


def test_overlap_removal_drops_fragments_only_when_requested() -> None:
    proposals = [(0.5, "2 3"), (0.9, "1 3")]
    assert disjoint(proposals, 6, 0) == ["1 3", "4 1"]
    assert disjoint(proposals, 6, 0.5) == ["1 3"]


def test_mask_confidence_changes_pixel_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    class Detector:
        def eval(self) -> "Detector":
            return self

        def __call__(self, _: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            return [
                {
                    "scores": torch.tensor([0.9, 0.8]),
                    "masks": torch.tensor([[[[0.6, 0.6, 0.0]]], [[[0.0, 0.9, 0.9]]]]),
                }
            ]

    monkeypatch.setattr(model, "load_image", lambda _: torch.zeros(3, 1, 3))
    ordinary, weighted = model.rankings(
        cast("MaskRCNN", Detector()),
        Path("image.png"),
        settings("E03"),
        torch.device("cpu"),
        weighted=True,
    )
    assert disjoint(ordinary, 3, 0.5) == ["1 2", "3 1"]
    assert disjoint(weighted, 3, 0.5) == ["2 2", "1 1"]


def test_group_split_and_oof_coverage(tmp_path: Path) -> None:
    images = tmp_path / "train"
    images.mkdir()
    with (tmp_path / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "annotation", "width", "height", "cell_type", "sample_id"))
        for index in range(14):
            image_id = str(index)
            content = str(index if index not in (1, 3) else 1)
            (images / f"{image_id}.png").write_text(content, encoding="ascii")
            writer.writerow((image_id, "1 1", 3, 2, "cell", index // 2))
    rows, folds, _, split_hash = dataset(tmp_path)
    by_id = dict(zip((row.id for row in rows), folds, strict=True))
    assert by_id["0"] == by_id["1"] == by_id["2"] == by_id["3"]
    assert dataset(tmp_path)[3] == split_hash
    path = tmp_path / "oof.json"
    saved = [
        {"id": row.id, "fold": fold, "masks": []} for row, fold in zip(rows, folds, strict=True)
    ]
    path.write_text(json.dumps(saved[:-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        read_rows(path, rows, folds)
    saved[1] = saved[0]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="OOF"):
        read_rows(path, rows, folds)


@pytest.mark.parametrize(
    "masks",
    [
        [[float("nan"), "1 1"]],
        [[float("inf"), "1 1"]],
        [[1.01, "1 1"]],
        [[-0.1, "1 1"]],
        [[True, "1 1"]],
        [[0.5]],
        [[0.5, 123]],
        [[0.5, "6 2"]],
        [[0.5, "3 1 1 1"]],
        [[0.5, "1 2"], [0.4, "1 1 2 1"]],
    ],
)
def test_invalid_raw_candidate(tmp_path: Path, masks: object) -> None:
    path = tmp_path / "raw.json"
    path.write_text(json.dumps([{"id": "image", "fold": 0, "masks": masks}]), encoding="utf-8")
    rows = [Image("image", path, 3, 2, "cell", "sample", [])]
    with pytest.raises(ValueError, match="OOF"):
        read_rows(path, rows, [0], kind="raw")


@pytest.mark.parametrize("fold", [1, False, "0"])
def test_wrong_oof_fold(tmp_path: Path, fold: object) -> None:
    path = tmp_path / "oof.json"
    path.write_text(json.dumps([{"id": "image", "fold": fold, "masks": []}]), encoding="utf-8")
    with pytest.raises(ValueError, match="OOF"):
        read_rows(path, [Image("image", path, 3, 2, "cell", "sample", [])], [0])


def test_offline_notebook_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        check_notebook_inference(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(previous_threads)


def check_notebook_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "input"
    test = root / "sartorius-cell-instance-segmentation" / "test"
    test.mkdir(parents=True)
    image = test / "fixed.png"
    PillowImage.fromarray(np.arange(64 * 80, dtype=np.uint8).reshape(64, 80)).save(image)
    weights = root / "model"
    weights.mkdir()
    device = torch.device("cpu")
    config = settings("E03")
    config.update(min_size=64, max_size=80, detections=5, score_threshold=0.0, mask_threshold=0.25)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        detector = build(config, pretrained=False)
    expected = model.predict(detector, image, config, device)
    assert expected
    torch.save(detector.state_dict(), weights / WEIGHTS)
    reloaded = build(config, pretrained=False)
    reloaded.load_state_dict(torch.load(weights / WEIGHTS, map_location=device, weights_only=True))
    assert model.weights_hash(detector) == model.weights_hash(reloaded)
    assert model.predict(reloaded, image, config, device) == expected
    write_notebook(weights, "owner", "owner/model", config)
    validate(weights)
    notebook = json.loads((weights / NOTEBOOK).read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        code = cell["source"].replace('"/kaggle/input"', repr(str(root)))
        exec(code.replace('torch.device("cuda")', 'torch.device("cpu")'), namespace)
    actual = eval("predict(model, paths[0], config, device)", namespace)
    assert actual == expected
    with (tmp_path / "submission.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["id", "predicted"],
            *[["fixed", rle] for rle in disjoint(expected, 64 * 80, config["retain"])],
        ]


@pytest.mark.parametrize("change", ["oof", "export_source", None])
def test_export_checks_oof_and_seals_export_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    path = tmp_path / "selection.toml"
    path.write_text(
        'base_trial = "E03"\ncandidates = ["S00-r2", "S01-r2"]\ntolerance = 0.001\n',
        encoding="utf-8",
    )
    (tmp_path / "quality-gates.toml").write_text("version = 1\n", encoding="utf-8")

    def source(trial: str, plan_path: Path) -> SourceSpec:
        return SourceSpec(tmp_path, ("selection.toml",), dict(train.run_settings(trial, plan_path)))

    rows = [Image(str(i), path, 3, 2, "cell", str(i), ["1 1"]) for i in range(3)]
    store = RunStore(tmp_path)
    monkeypatch.setattr(train, "RunStore", lambda *_: store)
    monkeypatch.setattr(train, "dataset", lambda: (rows, [0, 1, 2], "data", "split"))
    monkeypatch.setattr(train, "source", source)
    monkeypatch.setattr(train, "fit", lambda *_, **__: object())
    monkeypatch.setattr(
        train,
        "rankings",
        lambda *_, **__: ([(0.9, "1 2"), (0.8, "1 1")], [(0.6, "1 2"), (0.7, "1 1")]),
    )
    train.run("S00", "S00-r2", "S00-r2", path)
    run_path = train.run("S01", "S01-r2", "S00-r2", path)
    selection_plan = train.plan(path)
    receipt = tmp_path / "selection.json"
    selection = select_runs(store, selection_plan, train.assess)
    assert selection.winner == "S01-r2"
    trial = selected_trial(path, store.record(selection.winner))
    assert trial == "S01"
    write_selection(receipt, selection)
    entry = tmp_path / "export.py"
    entry.write_text("VERSION = 1\n", encoding="utf-8")
    if change == "oof":
        (run_path / train.PREDICTIONS).write_text("[]", encoding="utf-8")

    def write(directory: Path, _: object) -> None:
        if change == "oof":
            pytest.fail("改変された成果物で最終学習を開始しました")
        (directory / WEIGHTS).write_bytes(b"weights")
        if change == "export_source":
            entry.write_text("VERSION = 2\n", encoding="utf-8")

    def publish() -> Path:
        return publish_export(
            store,
            "F00",
            plan=selection_plan,
            receipt=receipt,
            assess=train.assess,
            source=source(trial, path),
            write=write,
            required_outputs=(WEIGHTS,),
            validate_outputs=lambda _: None,
            export_files=("export.py",),
        )

    if change is not None:
        with pytest.raises(ValueError):
            publish()
    else:
        evidence = json.loads((publish() / "export_evidence.json").read_text(encoding="utf-8"))
        assert set(json.loads(evidence["source"])["files"]) == {"selection.toml", "export.py"}
