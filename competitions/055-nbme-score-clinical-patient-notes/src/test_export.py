import csv
import json
from pathlib import Path
from zipfile import ZipFile

import model as model_module
import pytest
import torch
from data import load_pairs
from deployment import MODEL_ARCHIVE, validate
from export import write_notebook
from model import Tagger, decode_pair, encode, predict
from transformers import BertTokenizer, DistilBertConfig, DistilBertModel


def write_csv(path: Path, rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_expanded_notebook_matches_model_and_rejects_changed_file(tmp_path: Path) -> None:
    inputs = tmp_path / "input"
    data = inputs / "competition"
    model_dir = tmp_path / "serialized"
    package = tmp_path / "export"
    for directory in (data, model_dir, package):
        directory.mkdir(parents=True)
    write_csv(
        data / "patient_notes.csv",
        [("case_num", "pn_num", "pn_history"), ("0", "1", "hello world")],
    )
    write_csv(
        data / "features.csv",
        [("case_num", "feature_num", "feature_text"), ("0", "1", "hello"), ("0", "2", "world")],
    )
    write_csv(
        data / "test.csv",
        [("id", "case_num", "pn_num", "feature_num"), ("a", "0", "1", "1"), ("b", "0", "1", "2")],
    )
    write_csv(data / "sample_submission.csv", [("id", "location"), ("b", ""), ("a", "")])

    vocab = {
        token: i
        for i, token in enumerate(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"))
    }
    tok = BertTokenizer(vocab=vocab)
    tok.save_pretrained(model_dir)
    DistilBertConfig(
        vocab_size=len(vocab), n_layers=1, n_heads=2, dim=8, hidden_dim=16
    ).save_pretrained(model_dir)
    model = Tagger(str(model_dir))
    with torch.no_grad():
        model.head.bias.fill_(5)
    assert isinstance(tok.pad_token_id, int)
    pairs = load_pairs(data, training=False)
    scores = predict(
        model, encode(pairs, tok, "E03"), len(pairs), tok.pad_token_id, torch.device("cpu")
    )
    expected = {
        pair.id: "; ".join(f"{a} {b}" for a, b in decode_pair(pair, row, "E03"))
        for pair, row in zip(pairs, scores, strict=True)
    }
    torch.save(model.state_dict(), model_dir / "weights.pt")
    with ZipFile(package / MODEL_ARCHIVE, "w") as archive:
        for path in model_dir.iterdir():
            archive.write(path, path.name)
    write_notebook(package, "owner", "owner/weights", "E03")
    validate(package)
    with ZipFile(package / MODEL_ARCHIVE) as archive:
        archive.extractall(inputs / "dataset")

    notebook = json.loads((package / "submit.ipynb").read_text(encoding="utf-8"))
    output = tmp_path / "submission.csv"
    namespace: dict[str, object] = {}
    for cell in notebook["cells"][:-1]:
        exec(compile(cell["source"], cell["id"], "exec"), namespace)
    inference = notebook["cells"][-1]["source"]
    local = (
        inference.replace("'/kaggle/input'", repr(str(inputs)))
        .replace("'/kaggle/working/submission.csv'", repr(str(output)))
        .replace("torch.device('cuda')", "torch.device('cpu')")
    )
    exec(compile(local, "inference", "exec"), namespace)
    with output.open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["id", "location"],
            ["b", expected["b"]],
            ["a", expected["a"]],
        ]
    (inputs / "dataset" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"SHA-256.*tokenizer_config"):
        exec(compile(local, "inference", "exec"), namespace)


def test_fit_enables_training_on_pretrained_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = DistilBertModel(
        DistilBertConfig(vocab_size=8, n_layers=1, n_heads=2, dim=8, hidden_dim=16)
    ).eval()
    training_modes: list[bool] = []
    encoder.register_forward_pre_hook(lambda module, args: training_modes.append(module.training))
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda *args: encoder)
    window = model_module.Window(0, [2, 5, 3], [None, (0, 1), None], [-1, 1, -1])
    model_module.fit([window], 0, torch.device("cpu"))
    assert training_modes == [True] * model_module.EPOCHS
