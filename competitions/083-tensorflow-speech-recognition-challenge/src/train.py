"""事前登録した音声実験の学習・検査・成果物保存。"""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from dataset import CLASSES, OUTPUT, ROOT, load_prepared, make_folds, submission
from network import Features, environment, feature_batches, fit_model, predict, seed_all
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from supervision import FINE_CLASSES, fine_labels

from kumite.checks import validate_probabilities
from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore

EXPERIMENTS = {
    "E00": ("学習foldのクラス頻度", "定数予測の基準"),
    "E01": ("log-melと小型CNN", "音声表現の学習で定数予測を上回る"),
    "E02": ("log線形周波数へ変更", "周波数表現がコマンドの識別に影響する"),
    "E03": ("±100ms時間移動", "発話位置の変化への頑健性が改善する"),
    "E04": ("学習側背景音を混合", "雑音への頑健性が改善する"),
    "E05": (
        "unknownを20語へ細分化して31クラス学習",
        "語を識別する補助教師情報が12クラスAccuracyを改善する",
    ),
}


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float(accuracy_score(y, proba.argmax(axis=1)))


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    predicted = proba.argmax(axis=1)
    return dict(
        accuracy=accuracy(y, proba),
        macro_f1=float(f1_score(y, predicted, average="macro")),
        logloss=float(log_loss(y, proba, labels=np.arange(12))),
        recall=(
            np.diag(confusion_matrix(y, predicted, labels=np.arange(12)))
            / np.maximum(1, np.bincount(y, minlength=12))
        ).tolist(),
        confusion=confusion_matrix(y, predicted, labels=np.arange(12)).tolist(),
    )


def train(
    run_id: str,
    experiment: str,
    seed: int = 42,
    random_split: bool = False,
    benchmark: bool = False,
) -> Path:
    store = RunStore(ROOT)
    with store.reserve(run_id) as reservation:
        start = time.monotonic()
        config = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
        if experiment == "E05":
            registration = json.loads((ROOT / "p5_registration.json").read_text(encoding="utf-8"))
            if registration["experiment"] != "E05" or registration["parent"] != "E01":
                raise ValueError("E05の事前登録が不正です")
            for required in registration["registered_after_completed_runs"]:
                store.record(required)
        spent = sum(
            r.runtime_sec if r.runtime_sec is not None else r.params.get("budget_charge_seconds", 0)
            for r in store.records()
        )
        remaining = config["gpu_budget_seconds"] - spent
        if remaining <= 0:
            raise TimeoutError("実行時間予算を使い切りました")
        deadline = start + remaining
        frame, test, waves, test_waves, summary = load_prepared()
        # cacheの内容も学習前に照合する。
        for name in ("train", "test"):
            if file_sha256(OUTPUT / "prepared" / f"{name}.npy") != summary["caches"][name]:
                raise ValueError(f"{name}音声cacheのハッシュ不一致")
        folds = make_folds(frame, seed, random_split)
        splits = folds_to_splits(folds)
        if benchmark:
            splits = splits[:1]
        y = frame.label.to_numpy(dtype=np.int64)
        training_y = fine_labels(frame) if experiment == "E05" else y
        models = []
        seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = Features(linear=experiment == "E02").to(device)
        train_features = query_features = gpu_waves = None
        if experiment != "E00":
            gpu_waves = torch.tensor(np.array(waves), device=device)
            train_features = feature_batches(gpu_waves, extractor, device)
            query_features = feature_batches(test_waves, extractor, device)
        print(f"{run_id}: train={len(frame)} test={len(test)} device={device}", flush=True)

        def fit(train_idx: np.ndarray, valid_idx: np.ndarray) -> FoldOutput:
            k = len(models)
            if experiment == "E00":
                probabilities = np.bincount(y[train_idx], minlength=12) / len(train_idx)
                return FoldOutput(
                    np.tile(probabilities, (len(valid_idx), 1)),
                    np.tile(probabilities, (len(test), 1)),
                    {"constant": probabilities.tolist()},
                )
            assert (
                train_features is not None and query_features is not None and gpu_waves is not None
            )
            model, mean, std, losses = fit_model(
                train_features,
                gpu_waves,
                training_y,
                train_idx,
                extractor,
                experiment,
                seed + k,
                config["epochs"],
                config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                deadline=deadline,
            )
            valid = predict(
                model, train_features[torch.as_tensor(valid_idx, device=device)], mean, std
            )
            query = predict(model, query_features, mean, std)
            models.append(
                dict(
                    state_dict={
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    mean=mean.cpu(),
                    std=std.cpu(),
                    linear=experiment == "E02",
                )
            )
            overlap = len(set(frame.iloc[train_idx].group) & set(frame.iloc[valid_idx].group))
            info = dict(
                fold=k,
                epochs=config["epochs"],
                seed=seed + k,
                losses=losses,
                group_overlap=overlap,
                valid_metrics=metrics(y[valid_idx], valid),
                preprocessing=dict(
                    mean=mean.cpu().tolist(),
                    std=std.cpu().tolist(),
                    sample_rate=16000,
                    n_fft=512,
                    win_length=400,
                    hop_length=160,
                    bins=40,
                ),
                environment=environment(),
            )
            print(f"fold {k}: accuracy={accuracy(y[valid_idx], valid):.6f}", flush=True)
            return FoldOutput(valid, query, info)

        result = cross_fit_splits(len(frame), splits, fit)
        validated = np.concatenate([valid for _, valid in splits])
        validate_probabilities(result.oof[validated], len(validated), 12)
        validate_probabilities(result.query, len(test), 12)
        split_version = f"{'random' if random_split else 'speaker-background'}-3fold-v1-seed{seed}"
        if benchmark:
            split_version += "-benchmark-fold0"
        evaluation = Evaluation(
            ROOT.name,
            "accuracy",
            True,
            accuracy,
            summary["data_hash"],
            split_version,
            array_sha256(folds),
            seed,
        )
        changed, hypothesis = EXPERIMENTS[experiment]
        trial = Trial(
            hypothesis,
            changed,
            "class-frequency" if experiment == "E00" else "SmallCNN",
            "audio-v1",
            params=dict(
                experiment=experiment,
                config=config,
                benchmark=benchmark,
                random_split=random_split,
                classes=CLASSES,
                training_classes=FINE_CLASSES if experiment == "E05" else CLASSES,
            ),
        )

        def write(path: Path) -> None:
            submission(test, result.query).to_csv(path / "submission.csv", index=False)
            frame.assign(fold=folds).to_csv(path / "manifest.csv", index=False)
            test.to_csv(path / "test_manifest.csv", index=False)
            (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (path / "metrics.json").write_text(
                json.dumps(metrics(y[validated], result.oof[validated]), indent=2), encoding="utf-8"
            )
            for k, checkpoint in enumerate(models):
                torch.save(checkpoint, path / f"fold{k}.pt")
            with zipfile.ZipFile(path / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for source in sorted((ROOT / "src").glob("*.py")):
                    archive.write(source, source.name)
                for name in ("pyproject.toml", "uv.lock"):
                    archive.write(REPO_ROOT / name, name)

        return publish_cv_run(
            store,
            reservation,
            evaluation,
            trial,
            y,
            splits,
            result,
            time.monotonic() - start,
            write,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43], default=42)
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    train(args.run_id, args.experiment, args.seed, args.random_split, args.benchmark)
