"""固定epochのDistilBERT。モデルとtokenizerをcommit revisionで固定する。"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from kumite.crossfit import FoldOutput
from kumite.provenance import files_sha256

MODEL = "distilbert/distilbert-base-uncased"


def prepare(texts, test_texts):
    revision = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    directory = Path(
        snapshot_download(
            MODEL,
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "tokenizer*", "vocab.txt"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)

    def encode(values):
        return tokenizer(
            list(values), truncation=True, max_length=256, padding="max_length", return_tensors="pt"
        )

    identity = {
        "pretrained/revision": revision,
        "pretrained/files": files_sha256(sorted(f for f in directory.iterdir() if f.is_file())),
    }
    return {**encode(texts), "directory": str(directory)}, encode(test_texts), identity


def new_model(directory) -> DistilBertForSequenceClassification:
    model: DistilBertForSequenceClassification = cast(
        Any, DistilBertForSequenceClassification.from_pretrained
    )(directory, num_labels=6, problem_type="multi_label_classification", local_files_only=True)
    return model


def fit_transformer(encoded, encoded_test, y, tr, va, seed, epochs, *, max_steps=None):
    if not torch.cuda.is_available():
        raise RuntimeError("E05にはCUDA GPUが必要です")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = new_model(encoded["directory"])
    torch.nn.Module.cuda(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = math.ceil(len(tr) / 32) * epochs
    warmup = max(1, int(steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0, (steps - s) / max(1, steps - warmup)))
    )
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    completed = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for start in range(0, len(order), 32):
            idx = order[start : start + 32]
            batch = {k: encoded[k][idx].cuda() for k in ("input_ids", "attention_mask")}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    **batch, labels=torch.as_tensor(y[idx], dtype=torch.float32, device="cuda")
                )
            loss = cast(torch.Tensor, output.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            completed += 1
            if max_steps is not None and completed >= max_steps:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                del optimizer
                torch.cuda.empty_cache()
                return FoldOutput(
                    np.empty((0, 6)),
                    np.empty((0, 6)),
                    {
                        "seconds": elapsed,
                        "steps": completed,
                        "seconds_per_step": elapsed / completed,
                    },
                )
        print(f"epoch {epoch + 1}/{epochs}: {time.monotonic() - started:.1f}s", flush=True)

    def predict(data, indices):
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(indices), 64):
                idx = indices[start : start + 64]
                batch = {k: data[k][idx].cuda() for k in ("input_ids", "attention_mask")}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = cast(torch.Tensor, model(**batch).logits)
                values.append(logits.float().sigmoid().cpu().numpy())
        return np.concatenate(values)

    valid, test = (
        predict(encoded, va),
        predict(encoded_test, np.arange(len(encoded_test["input_ids"]))),
    )
    info = {
        "epochs": epochs,
        "train_steps": completed,
        "max_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    del optimizer
    torch.cuda.empty_cache()
    return FoldOutput(valid, test, info)


def cpu_smoke():
    from transformers import DistilBertConfig

    config = DistilBertConfig(
        vocab_size=32,
        max_position_embeddings=16,
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        problem_type="multi_label_classification",
    )
    config.num_labels = 6
    model = DistilBertForSequenceClassification(config)
    before = model.classifier.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = model(
        input_ids=torch.ones((2, 8), dtype=torch.long),
        attention_mask=torch.ones((2, 8), dtype=torch.long),
        labels=torch.zeros((2, 6)),
    )
    loss = cast(torch.Tensor, result.loss)
    loss.backward()
    opt.step()
    assert result.logits.shape == (2, 6) and torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.benchmark:
        from cv import make_folds
        from dataset import COMPETITION_DIR, LABELS, load_data

        train, test, _ = load_data()
        y = train[list(LABELS)].to_numpy(dtype=np.int64)
        folds = make_folds(train.comment_text, y)
        encoded, encoded_test, identity = prepare(train.comment_text, test.comment_text)
        out = fit_transformer(
            encoded,
            encoded_test,
            y,
            np.flatnonzero(folds != 0),
            np.flatnonzero(folds == 0),
            42,
            2,
            max_steps=40,
        )
        estimate = out.info["seconds_per_step"] * math.ceil((folds != 0).sum() / 32) * 2 * 3 * 1.4
        epochs = 2 if estimate <= 4 * 3600 else 1 if estimate / 2 <= 4 * 3600 else 0
        report: dict[str, Any] = {
            **out.info,
            "estimated_2epoch_seconds": estimate,
            "epochs": epochs,
            **identity,
        }
        dest = COMPETITION_DIR / "outputs" / "benchmark.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        cpu_smoke()
