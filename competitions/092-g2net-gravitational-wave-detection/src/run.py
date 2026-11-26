"""実データの検査、交差検証、選定と提出生成。"""

import argparse
import csv
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import numpy as np
import torch
from dataset import SEED, fingerprint, make_splits, read_ids, read_waveform, waveform_path
from model import BAND_MASK, SCALE, build_model, fit_whitening, tensor, transform
from sklearn.metrics import roc_auc_score
from torch import nn

from kumite.crossfit import FoldOutput, cross_fit_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    dense_oof,
    publish_export,
    reserve_cv,
    select_runs,
    verify_selection,
    write_selection,
)
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
BATCH = 256
EPOCHS = 3
SUBMISSION = "submission.csv"


@dataclass(frozen=True)
class Config:
    channels: int
    bandpass: bool
    whiten: bool
    change: str


CONFIGS = {
    "B00": Config(1, False, False, "固定倍率の Hanford 波形"),
    "E01": Config(1, True, False, "25–500 Hz の帯域制限"),
    "E02": Config(1, True, True, "fold 学習波形で推定した PSD による白色化"),
    "E03": Config(3, True, True, "3 検出器の波形を入力"),
}


def score(y: np.ndarray, predictions: np.ndarray) -> float:
    if predictions.shape != y.shape or not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("予測確率が不正です")
    return float(roc_auc_score(y, predictions))


def inspect(
    data: Path, train: list[str], test: list[str], *, diagnostics: bool = False
) -> tuple[np.ndarray, str, dict[str, np.ndarray]]:
    if set(train).intersection(test):
        raise ValueError("学習とテストの ID が重複しています")
    digest = hashlib.sha256()
    groups: list[str] = []
    rms_by_split: dict[str, np.ndarray] = {}
    train_hashes: set[str] = set()
    for split, ids in (("train", train), ("test", test)):
        seen: set[str] = set()
        rms = np.empty((len(ids), 3)) if diagnostics else None
        constant = np.zeros(3, dtype=int)
        band_fraction = []
        for index, key in enumerate(ids):
            wave = read_waveform(waveform_path(data, split, key))
            checksum = fingerprint(wave)
            digest.update(f"{split}/{key}:{checksum}\n".encode())
            seen.add(checksum)
            if rms is not None:
                scaled = wave * SCALE
                rms[index] = np.sqrt(np.mean(np.square(scaled), axis=1))
                constant += np.all(wave == wave[:, :1], axis=1)
                if int(checksum[:4], 16) % 64 == 0:
                    power = np.square(np.abs(np.fft.rfft(scaled)))
                    band_fraction.append(
                        power[:, BAND_MASK].sum(axis=1) / np.maximum(power.sum(axis=1), 1e-30)
                    )
            if split == "train":
                groups.append(checksum)
        if split == "train":
            train_hashes = seen
        else:
            print(f"学習とテストの完全重複: {len(seen & train_hashes)} 種類")
        print(f"{split}: {len(ids)} 行、完全重複 {len(ids) - len(seen)} 行")
        if rms is not None:
            rms_by_split[split] = rms
            print(
                f"定数チャネル: {constant}、RMS の 10/50/90% 点: {np.quantile(rms, [0.1, 0.5, 0.9], axis=0)}"
            )
            if band_fraction:
                print(f"帯域内パワーの割合（中央値）: {np.median(band_fraction, axis=0)}")
    return np.array(groups), digest.hexdigest(), rms_by_split


def batches(paths: list[Path], indices: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, len(indices), BATCH):
        rows = indices[start : start + BATCH]
        yield rows, np.stack([read_waveform(paths[int(index)]) for index in rows])


def fit_fold(
    fold: int,
    train: np.ndarray,
    valid: np.ndarray,
    paths: list[Path],
    test: list[Path],
    y: np.ndarray,
    config: Config,
    device: torch.device,
    *,
    epochs: int = EPOCHS,
) -> FoldOutput:
    torch.manual_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)
    model = build_model(config.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss()
    whitening = fit_whitening(wave for _, wave in batches(paths, train)) if config.whiten else None
    for epoch in range(epochs):
        start = time.monotonic()
        model.train()
        for rows, wave in batches(paths, rng.permutation(train)):
            optimizer.zero_grad()
            logits = model(
                tensor(transform(wave, config.channels, config.bandpass, whitening), device)
            )
            loss(logits[:, 0], tensor(y[rows].astype(np.float32), device)).backward()
            optimizer.step()
        print(f"fold {fold} epoch {epoch + 1}: {time.monotonic() - start:.1f} 秒", flush=True)

    @torch.inference_mode()
    def predict(files: list[Path], rows: np.ndarray) -> np.ndarray:
        predictions = []
        model.eval()
        for _, wave in batches(files, rows):
            inputs = tensor(transform(wave, config.channels, config.bandpass, whitening), device)
            predictions.append(model(inputs)[:, 0].sigmoid().cpu().numpy())
        return np.concatenate(predictions)

    return FoldOutput(
        predict(paths, valid), predict(test, np.arange(len(test))), {"epochs": epochs}
    )


def source(config: Config, plan_path: Path) -> SourceSpec:
    files = [
        *(ROOT / "src" / name for name in ("dataset.py", "model.py", "run.py")),
        ROOT / "quality-gates.toml",
        plan_path,
    ]
    modules = (
        "__init__",
        "crossfit",
        "cv_run",
        "experiments",
        "paths",
        "provenance",
        "quality_gates",
        "runs",
        "source_audit",
    )
    files.extend(REPO_ROOT / "src/kumite" / f"{name}.py" for name in modules)
    files.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"))
    return SourceSpec(
        REPO_ROOT,
        tuple(file.relative_to(REPO_ROOT).as_posix() for file in files),
        {"config": asdict(config), "epochs": EPOCHS, "batch": BATCH, "seed": SEED},
        import_roots=("src", (ROOT / "src").relative_to(REPO_ROOT).as_posix()),
    )


def write_csv(path: Path, ids: list[str], predictions: np.ndarray) -> None:
    if predictions.shape != (len(ids),) or not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("提出用の確率が不正です")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "target"))
        writer.writerows(zip(ids, predictions.tolist(), strict=True))


def validate_csv(path: Path, ids: list[str]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["id", "target"]:
            raise ValueError("提出の列が不正です")
        for key, (identifier, value) in zip(ids, reader, strict=True):
            if key != identifier or not 0 <= float(value) <= 1:
                raise ValueError("提出の ID 順または確率が不正です")


def analyze(y: np.ndarray, predictions: np.ndarray, rms: np.ndarray) -> None:
    for detector, noise in zip(("Hanford", "Livingston", "Virgo"), rms.T, strict=True):
        edges = np.unique(np.quantile(noise, [0.25, 0.5, 0.75]))
        for index in range(len(edges) + 1):
            rows = np.searchsorted(edges, noise, side="right") == index
            labels = y[rows]
            if len(np.unique(labels)) == 2:
                print(
                    f"{detector} RMS 区間 {index + 1}: {len(labels)} 行、AUC {score(labels, predictions[rows]):.5f}"
                )


def cuda_device() -> torch.device:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が必要です")
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda")


def smoke(train: list[str], y: np.ndarray, test: list[str]) -> None:
    paths = [waveform_path(DATA, "train", key) for key in train[: 5 * BATCH]]
    queries = [waveform_path(DATA, "test", key) for key in test[:BATCH]]
    device = cuda_device()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    result = fit_fold(
        0,
        np.arange(4 * BATCH),
        np.arange(4 * BATCH, 5 * BATCH),
        paths,
        queries,
        y,
        CONFIGS["B00"],
        device,
        epochs=1,
    )
    for probabilities in (result.valid, result.query):
        if probabilities.shape != (BATCH,) or not np.all(
            (probabilities >= 0) & (probabilities <= 1)
        ):
            raise ValueError("疎通試験の予測確率が不正です")
    print(
        f"疎通試験: {time.monotonic() - start:.1f} 秒、CUDA ピーク {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "inspect", "cv", "select", "export", "analyze"))
    parser.add_argument("plan", choices=("baseline", "p4", "p6"), nargs="?", default="p4")
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args(argv)
    train, y = read_ids(DATA / "training_labels.csv", training=True)
    test, _ = read_ids(DATA / "sample_submission.csv")
    if args.action == "smoke":
        smoke(train, y, test)
        return
    groups, wave_hash, rms = inspect(
        DATA, train, test, diagnostics=args.action in {"inspect", "analyze"}
    )
    splits = make_splits(y, groups)
    data_hash = hashlib.sha256(f"{wave_hash}:{array_sha256(y)}".encode()).hexdigest()
    split_hash = array_sha256(split_roles(len(y), splits))
    if args.action == "inspect":
        OUTPUTS.mkdir(exist_ok=True)
        np.savez_compressed(
            OUTPUTS / "data-manifest.npz",
            train_ids=np.array(train),
            test_ids=np.array(test),
            groups=groups,
            roles=split_roles(len(y), splits),
            data_hash=data_hash,
            split_hash=split_hash,
            train_rms=rms["train"],
            test_rms=rms["test"],
        )
        print(f"data={data_hash} split={split_hash}")
        return
    plan_path = ROOT / ("selection-p6.json" if args.plan == "p6" else "selection-plans.json")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))[args.plan]
    plan = SelectionPlan(tuple(plan_data["candidates"]), plan_data["tolerance"])
    configs = {**CONFIGS, "B00-initial": CONFIGS["B00"]}
    if args.plan == "p6":
        configs.update({key: Config(**settings) for key, settings in plan_data["configs"].items()})
    store = RunStore(ROOT)
    OUTPUTS.mkdir(exist_ok=True)
    receipt = OUTPUTS / f"selection-{args.plan}.json"
    assess = dense_oof(y, splits, score, expected_shape=y.shape, data_hash=data_hash)
    if args.action == "cv":
        if args.run_id not in plan.candidates:
            parser.error("選定計画に含まれる run ID を指定してください")
        config = configs[args.run_id]
        spec = source(config, plan_path)
        paths = [waveform_path(DATA, "train", key) for key in train]
        queries = [waveform_path(DATA, "test", key) for key in test]
        device = cuda_device()
        with reserve_cv(store, args.run_id, spec, {}, plan) as (reservation, start):
            initial = time.monotonic()
            folds = count()
            result = cross_fit_splits(
                len(y),
                splits,
                lambda tr, va: fit_fold(next(folds), tr, va, paths, queries, y, config, device),
            )
            if result.query.shape != (len(test),) or not np.all(
                (result.query >= 0) & (result.query <= 1)
            ):
                raise ValueError("テストの予測確率が不正です")
            publish_cv_run(
                store,
                reservation,
                Evaluation("092", "roc_auc", True, score, data_hash, "v1", split_hash, SEED),
                Trial(config.change, config.change, "1D CNN", "official-v1", asdict(config)),
                y,
                splits,
                result,
                time.monotonic() - initial,
                lambda _: None,
                gate=(spec, start, y.shape),
            )
    elif args.action == "select":
        write_selection(receipt, select_runs(store, plan, assess))
    elif args.action == "analyze":
        winner = verify_selection(store, plan, receipt, assess).winner
        analyze(y, store.load_array(winner, "oof.npy"), rms["train"])
    else:
        if args.run_id is None:
            parser.error("export ID を指定してください")
        winner = verify_selection(store, plan, receipt, assess).winner
        config = configs[winner]
        publish_export(
            store,
            args.run_id,
            plan=plan,
            receipt=receipt,
            assess=assess,
            source=source(config, plan_path),
            write=lambda path, run: write_csv(
                path / SUBMISSION, test, store.load_array(run.run_id, "test.npy")
            ),
            required_outputs=(SUBMISSION,),
            validate_outputs=lambda path: validate_csv(path / SUBMISSION, test),
        )


if __name__ == "__main__":
    main()
