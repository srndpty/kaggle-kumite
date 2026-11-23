"""固定 CV、モデル選定、提出ファイル作成。"""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from dataset import DATA, OUTPUT, ROOT, SEED, SPLIT_VERSION, prepare
from model import BATCH_SIZE, N_CLASSES, Images, class_weights, ensemble, network, predict
from torch import nn
from torch.utils.data import DataLoader

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.provenance import file_sha256, files_sha256
from kumite.runs import RunStore

STORE = RunStore(ROOT)
EXPERIMENTS = {
    "E00": ("最多クラスを予測する", "定数"),
    "E01": ("葉画像から病斑を識別できる", "ResNet-18"),
    "E02": ("少数クラスの再現率を改善する", "クラス重み"),
    "E03": ("葉の向きへの依存を減らす", "学習時水平反転"),
    "E04": ("細かな病斑をより捉える", "320 px"),
    "E05": ("反転で変わる予測を平均すると精度が上がる", "推論時水平反転"),
}


def accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(y == probabilities.argmax(axis=1)))


def cv(trial: str, run_id: str, baseline: str | None = None) -> None:
    if (trial == "E05") != (baseline is not None):
        raise ValueError("E05 だけが --baseline を必要とします")
    split_file = OUTPUT / f"{SPLIT_VERSION}.csv"
    rows = pd.read_csv(split_file, dtype={"image_id": str})
    official = pd.read_csv(DATA / "train.csv", dtype={"image_id": str})
    official = official.sort_values("image_id", ignore_index=True)
    if not rows[["image_id", "label"]].equals(official[["image_id", "label"]]):
        raise ValueError("公式学習ラベルが分割の作成時と一致しません")
    if (rows.groupby("group")["fold"].nunique() != 1).any() or set(rows["fold"]) != {0, 1, 2}:
        raise ValueError("近重複グループまたは固定 fold が不正です")
    paths = [DATA / "train_images" / str(name) for name in rows["image_id"]]
    labels = rows["label"].to_numpy(dtype=np.int64)
    folds = rows["fold"].to_numpy(dtype=np.int64)
    evaluation = Evaluation(
        "059",
        "accuracy",
        True,
        accuracy,
        files_sha256([DATA / "train.csv", *paths]),
        SPLIT_VERSION,
        file_sha256(split_file),
        SEED,
    )
    size = 320 if trial == "E04" else 224
    hypothesis, change = EXPERIMENTS[trial]
    names = [f"fold-{i}.pt" for i in range(3)]
    inputs: dict[str, str] = {}
    if baseline is not None:
        source = STORE.record(baseline)
        if (
            source.params["trial"] != "E01"
            or source.split_hash != evaluation.split_hash
            or source.data_hash != evaluation.data_hash
        ):
            raise ValueError("再利用する E01 のデータまたは分割が一致しません")
        inputs = STORE.input_hashes(baseline, *names)
    configuration = Trial(
        hypothesis,
        change,
        "majority" if trial == "E00" else "resnet18",
        "official",
        {
            "trial": trial,
            "size": size,
            "epochs": 5,
            "batch_size": BATCH_SIZE,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "weights": "IMAGENET1K_V1",
            "flip": trial == "E05",
            "model_source_sha256": file_sha256(Path(__file__).with_name("model.py")),
        },
        inputs,
    )
    device = torch.device("cpu" if trial == "E00" else "cuda")
    start = perf_counter()
    with STORE.reserve(run_id) as reservation, TemporaryDirectory(dir=OUTPUT) as temporary:
        checkpoints: list[Path] = []

        def fit(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
            target = labels[train]
            if trial == "E00":
                most_common = int(np.bincount(target, minlength=N_CLASSES).argmax())
                probability = np.zeros((len(valid), N_CLASSES), dtype=np.float32)
                probability[:, most_common] = 1
            elif baseline is not None:
                name = names[int(folds[valid[0]])]
                checkpoint = Path(temporary) / name
                shutil.copyfile(STORE.artifact_path(baseline, name), checkpoint)
                checkpoints.append(checkpoint)
                probability = ensemble(
                    [checkpoint], [paths[i] for i in valid], size, device, flip=True
                )
            else:
                torch.manual_seed(SEED + len(checkpoints))
                model = network(pretrained=True).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
                weight = class_weights(target).to(device) if trial == "E02" else None
                loss_fn = nn.CrossEntropyLoss(weight=weight)
                scaler = torch.amp.GradScaler("cuda")
                loader = DataLoader(
                    Images([paths[i] for i in train], target.tolist(), size, trial == "E03"),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True,
                )
                for epoch in range(5):
                    model.train()
                    for images, truth in loader:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda"):
                            loss = loss_fn(model(images.to(device)), truth.to(device))
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    print(f"{trial} fold {len(checkpoints)} epoch {epoch + 1}/5", flush=True)
                checkpoint = Path(temporary) / f"fold-{len(checkpoints)}.pt"
                torch.save(model.state_dict(), checkpoint)
                checkpoints.append(checkpoint)
                probability = predict(model, [paths[i] for i in valid], size, device)
            validate_probabilities(probability, len(valid), N_CLASSES)
            print(f"fold {int(folds[valid[0]])} 推論完了", flush=True)
            return FoldOutput(probability, np.empty((0, N_CLASSES)))

        result = cross_fit(folds, fit)

        def write(path: Path) -> None:
            for checkpoint in checkpoints:
                checkpoint.replace(path / checkpoint.name)

        publish_cv_run(
            STORE,
            reservation,
            evaluation,
            configuration,
            labels,
            folds_to_splits(folds),
            result,
            perf_counter() - start,
            write,
            save_query=False,
        )


def select(run_ids: list[str], output: Path) -> None:
    split = OUTPUT / f"{SPLIT_VERSION}.csv"
    labels = pd.read_csv(split)["label"].to_numpy()
    candidates = []
    split_hash = file_sha256(split)
    data_hashes: set[str] = set()
    for run_id in run_ids:
        record = STORE.record(run_id)
        trial = str(record.params["trial"])
        if (
            trial not in EXPERIMENTS
            or trial == "E00"
            or record.split_hash != split_hash
            or record.split_version != SPLIT_VERSION
            or record.metric != "accuracy"
        ):
            raise ValueError(f"{run_id} は選定対象の分割・実験と一致しません")
        data_hashes.add(record.data_hash)
        oof = STORE.load_array(run_id, "oof.npy")
        validate_probabilities(oof, len(labels), N_CLASSES)
        score = accuracy(labels, oof)
        if not np.isclose(score, record.oof_score, rtol=0, atol=1e-12):
            raise ValueError(f"{run_id} の OOF 再採点が記録と一致しません")
        candidates.append((-score, trial, run_id))
    if len(data_hashes) != 1:
        raise ValueError("選定対象のデータが一致しません")
    _, trial, run_id = min(candidates)
    record = STORE.record(run_id)
    selection = {
        "run_id": run_id,
        "trial": trial,
        "size": record.params["size"],
        "flip": record.params.get("flip", False),
        "split_hash": record.split_hash,
        "weights": STORE.input_hashes(run_id, *(f"fold-{i}.pt" for i in range(3))),
    }
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def selected_weights(selection_file: Path) -> tuple[dict[str, object], list[Path]]:
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    run_id = str(selection["run_id"])
    record = STORE.record(run_id)
    if (
        selection["trial"] != record.params["trial"]
        or selection["size"] != record.params["size"]
        or selection["flip"] != record.params.get("flip", False)
        or selection["split_hash"] != record.split_hash
        or record.params["model_source_sha256"] != file_sha256(Path(__file__).with_name("model.py"))
    ):
        raise ValueError("選定記録と学習設定が一致しません")
    names = [f"fold-{i}.pt" for i in range(3)]
    if selection["weights"] != STORE.input_hashes(run_id, *names):
        raise ValueError("選定記録と重みが一致しません")
    return selection, [STORE.artifact_path(run_id, name) for name in names]


def final(selection_file: Path, output: Path) -> None:
    selection, checkpoints = selected_weights(selection_file)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if list(sample.columns) != ["image_id", "label"] or not sample["image_id"].is_unique:
        raise ValueError("サンプル提出の列または ID が不正です")
    paths = [DATA / "test_images" / str(name) for name in sample["image_id"]]
    probability = ensemble(
        checkpoints,
        paths,
        int(str(selection["size"])),
        torch.device("cpu"),
        selection["flip"] is True,
    )
    validate_probabilities(probability, len(sample), N_CLASSES)
    sample["label"] = probability.argmax(axis=1)
    sample.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    cv_parser = sub.add_parser("cv")
    cv_parser.add_argument("trial", choices=EXPERIMENTS)
    cv_parser.add_argument("--run-id", required=True)
    cv_parser.add_argument("--baseline")
    selection = sub.add_parser("select")
    selection.add_argument("run_ids", nargs="+")
    selection.add_argument("--output", type=Path, default=ROOT / "selection.json")
    submission = sub.add_parser("final")
    submission.add_argument("selection", type=Path)
    submission.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare())
    elif args.command == "cv":
        cv(args.trial, args.run_id, args.baseline)
    elif args.command == "select":
        select(args.run_ids, args.output)
    else:
        final(args.selection, args.output)


if __name__ == "__main__":
    main()
