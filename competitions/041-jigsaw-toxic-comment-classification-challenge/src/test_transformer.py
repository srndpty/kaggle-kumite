"""本番の学習経路で外側ラベルの独立性を検査する。モデル取得・GPUは不要。"""

import json

import numpy as np
import pytest
import torch
import transformer
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from kumite.checks import check_label_independence


@pytest.fixture
def tiny_training(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    models = []
    initial = []

    def factory(_directory):
        config = DistilBertConfig(
            vocab_size=32,
            max_position_embeddings=8,
            n_layers=1,
            n_heads=2,
            dim=16,
            hidden_dim=32,
            problem_type="multi_label_classification",
        )
        config.num_labels = 6
        model = DistilBertForSequenceClassification(config)
        models.append(model)
        initial.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return model

    monkeypatch.setattr(transformer, "new_model", factory)
    ids = torch.tensor([[2, i + 3, 0, 0] for i in range(6)])
    encoded = {"directory": "unused", "input_ids": ids, "attention_mask": ids.ne(0).long()}
    query = {k: v[:2] for k, v in encoded.items() if k != "directory"}
    try:
        yield encoded, query, models, initial
    finally:
        torch.set_num_threads(previous_threads)


def test_fit_uses_only_training_labels(tiny_training):
    encoded, query, models, initial = tiny_training
    y = np.random.default_rng(7).integers(0, 2, (6, 6))
    tr, va = np.array([0, 2, 4]), np.array([1, 3])
    outputs = []

    def build(labels):
        out = transformer.fit_transformer(
            encoded, query, labels, tr, va, 42, 3, device=torch.device("cpu")
        )
        outputs.append(out)
        state = models[-1].state_dict()
        assert any(not torch.equal(state[k], initial[-1][k]) for k in state)
        return [out.valid, out.query, *[v.detach().numpy().copy() for v in state.values()]]

    changed = y.copy()
    changed[[1, 3, 5]] = 1 - changed[[1, 3, 5]]
    check_label_independence(build, y, changed)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            models[0].state_dict().values(), models[1].state_dict().values(), strict=True
        )
    )
    # 正の対照: 学習ラベルを変えるとモデルは実際に更新方向を変える。
    changed_train = y.copy()
    changed_train[tr] = 1 - changed_train[tr]
    build(changed_train)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            models[0].state_dict().values(), models[2].state_dict().values(), strict=True
        )
    )
    assert outputs[0].valid.shape == (2, 6) and outputs[0].query.shape == (2, 6)
    environment = json.loads(json.dumps(outputs[0].info))["runtime_environment"]
    assert environment["device"] == "cpu" and environment["gpu_name"] is None
    assert environment["torch_version"] == str(torch.__version__)
    assert environment["transformers_version"] == transformer.transformers_version
    assert environment["cuda_build_version"] == torch.version.cuda
    assert environment["deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()


def test_default_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        transformer.fit_transformer({}, {}, np.zeros((2, 6)), np.array([0]), np.array([1]), 42, 1)


def test_runtime_environment_preserves_determinism_settings(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fixture GPU")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        info = transformer.runtime_environment(torch.device("cuda"))
        assert info["gpu_name"] == "fixture GPU"
        assert info["cuda_build_version"] == torch.version.cuda
        assert info["transformers_version"] == transformer.transformers_version
        assert info["deterministic_algorithms"] is True
        assert info["deterministic_warn_only"] is True
        assert info["cublas_workspace_config"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        json.dumps(info)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
