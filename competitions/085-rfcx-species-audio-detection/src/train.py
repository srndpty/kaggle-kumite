"""固定分割、OOF の公開、選定、提出 CSV の export。"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from dataset import ID, RATE, SPECIES, WINDOW, read_events, training_examples
from model import fit, predict

from kumite.crossfit import FoldOutput, cross_fit_splits, folds_to_splits, split_roles
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.group_folds import multilabel_group_folds
from kumite.lwlrap import lwlrap
from kumite.paths import REPO_ROOT
from kumite.provenance import array_sha256, files_sha256
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

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
SEED = 42
UPDATES = 800
TRIALS = {
    "E00": (False, WINDOW, 1, "真陽性窓と弱い背景負例で録音中の種を予測できる"),
    "E01": (True, WINDOW, 1, "誤検出を種別の負例として使うと種の順位が改善する"),
    "E02": (False, WINDOW // 2, 1, "重複走査は短い鳴き声の順位を改善する"),
    "E03": (False, WINDOW, 3, "上位３窓平均で単発誤検出の影響を抑えられる"),
    "E04": (False, WINDOW, 1, "周波数位置を明示すると種の順位が改善する"),
}
SHARED = (
    "__init__",
    "audio",
    "crossfit",
    "cv_run",
    "experiments",
    "group_folds",
    "lwlrap",
    "paths",
    "provenance",
    "quality_gates",
    "runs",
    "source_audit",
)


def pcm_fingerprint(path: Path) -> str:
    # FLAC STREAMINFO の PCM MD5 と音声形式によりコンテナのタグが異なる重複も束ねる。
    with path.open("rb") as stream:
        header = stream.read(42)
    if len(header) != 42 or header[:4] != b"fLaC" or header[5:8] != b"\x00\x00\x22":
        raise ValueError(f"不正な FLAC STREAMINFO: {path}")
    if not any(header[26:42]):
        raise ValueError(f"PCM MD5 が保存されていません: {path}")
    return header[18:42].hex()


def load_cv():
    originals = sorted((DATA / "train").glob("*.flac"))
    keys = [path.stem for path in originals]
    paths = [DATA / "pcm" / "train" / f"{key}.wav" for key in keys]
    annotation_files = [DATA / "train_tp.csv", DATA / "train_fp.csv"]
    true, false = map(read_events, annotation_files)
    if set(keys) != set(true) | set(false):
        raise ValueError("学習録音の ID と注釈の集合が一致しません")
    groups = [pcm_fingerprint(path) for path in originals]
    info = [sf.info(path) for path in paths]
    if any(item.samplerate != RATE or item.channels != 1 for item in info):
        raise ValueError("学習用 PCM は 32 kHz モノラルが必要です")
    lengths = np.array([item.frames for item in info])
    labels = np.zeros((len(keys), len(SPECIES)), dtype=np.bool_)
    for row, key in enumerate(keys):
        for event in true.get(key, []):
            labels[row, event.species] = True
    splits = folds_to_splits(multilabel_group_folds(groups, labels, n_folds=3, seed=SEED))
    if not all(labels[valid].any() for _, valid in splits):
        raise ValueError("陽性ラベルのない fold があります")
    evaluation = Evaluation(
        "085",
        "lwlrap",
        True,
        lwlrap,
        files_sha256([*annotation_files, DATA / "pcm" / "decoder.txt", *originals, *paths]),
        "v1",
        array_sha256(split_roles(len(labels), splits)),
        SEED,
    )
    return paths, keys, lengths, groups, true, false, labels, splits, evaluation


def source(plan_file: Path, trial: str) -> SourceSpec:
    files = [
        Path(__file__),
        COMP / "src/dataset.py",
        COMP / "src/model.py",
        COMP / "src/prepare.py",
        COMP / "quality-gates.toml",
        plan_file,
    ]
    return SourceSpec(
        REPO_ROOT,
        (
            *(path.relative_to(REPO_ROOT).as_posix() for path in files),
            *(f"src/kumite/{name}.py" for name in SHARED),
            "pyproject.toml",
            "uv.lock",
        ),
        {
            "trial": trial,
            "config": list(TRIALS[trial][:3]),
            "updates": UPDATES,
            "seed": SEED,
            "frequency_position": trial == "E04",
        },
        import_roots=("src", (COMP / "src").relative_to(REPO_ROOT).as_posix()),
    )


def test_paths() -> list[Path]:
    with (DATA / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [ID, *SPECIES]:
            raise ValueError("公式提出サンプルの列が不正です")
        ids = [row[ID] for row in reader]
    paths = [DATA / "test" / f"{key}.flac" for key in ids]
    if not ids or len(set(ids)) != len(ids) or set(paths) != set((DATA / "test").glob("*.flac")):
        raise ValueError("テスト録音の ID が公式サンプルと一致しません")
    return [DATA / "pcm" / "test" / f"{key}.wav" for key in ids]


def check_scores(scores: np.ndarray, count: int) -> None:
    if (
        scores.shape != (count, len(SPECIES))
        or not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all()
    ):
        raise ValueError("予測の形状または値が不正です")


def write_submission(path: Path, ids: list[str], scores: np.ndarray) -> None:
    check_scores(scores, len(ids))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([ID, *SPECIES])
        writer.writerows([key, *row] for key, row in zip(ids, scores, strict=True))


def validate_submission(path: Path, ids: list[str]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        header, *rows = csv.reader(stream)
    if header != [ID, *SPECIES] or [row[0] for row in rows if row] != ids:
        raise ValueError("提出 CSV の ID または列が不正です")
    check_scores(np.asarray([row[1:] for row in rows], dtype=float), len(ids))


def run(
    trial: str, run_id: str, plan_file: Path, plan: SelectionPlan, device: torch.device
) -> None:
    started = perf_counter()
    paths, keys, lengths, _, true, false, labels, splits, evaluation = load_cv()
    query = test_paths()
    hard, hop, top, hypothesis = TRIALS[trial]
    store, spec = RunStore(COMP), source(plan_file, trial)

    def fit_fold(train: np.ndarray, valid: np.ndarray) -> FoldOutput:
        pools = training_examples(keys, lengths, train, true, false, SEED, hard)
        network = fit(paths, pools, UPDATES, SEED, device, frequency_position=trial == "E04")
        valid_scores = np.stack([predict(network, paths[i], hop, top, device) for i in valid])
        query_scores = np.stack([predict(network, path, hop, top, device) for path in query])
        check_scores(valid_scores, len(valid))
        check_scores(query_scores, len(query))
        print(f"fold 完了: {lwlrap(labels[valid], valid_scores):.6f}", flush=True)
        return FoldOutput(valid_scores, query_scores, {"pools": [len(pool) for pool in pools]})

    def write(path: Path) -> None:
        (path / "test_ids.json").write_text(
            json.dumps([item.stem for item in query]), encoding="utf-8"
        )

    with reserve_cv(store, run_id, spec, {}, plan) as (reservation, start):
        result = cross_fit_splits(len(labels), splits, fit_fold)
        publish_cv_run(
            store,
            reservation,
            evaluation,
            Trial(
                hypothesis,
                trial,
                "log-mel SmallCNN",
                "official-v1",
                {"trial": trial, "updates": UPDATES, "test_hash": files_sha256(query)},
            ),
            labels,
            splits,
            result,
            perf_counter() - started,
            write,
            gate=(spec, start, labels.shape),
        )


def eda() -> None:
    _, keys, _, groups, true, false, labels, splits, _ = load_cv()
    lengths = [(event.end - event.start) / RATE for events in true.values() for event in events]
    overlapping = sum(
        any(
            t.species == f.species and f.start < t.end and t.start < f.end
            for t in true.get(key, [])
        )
        for key, events in false.items()
        for f in events
    )
    print(
        json.dumps(
            {
                "recordings": len(keys),
                "pcm_groups": len(set(groups)),
                "positives_by_species": labels.sum(axis=0).tolist(),
                "tp_seconds_quartiles": np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1])
                .round(4)
                .tolist(),
                "multilabel_recordings": int((labels.sum(axis=1) > 1).sum()),
                "fp_annotations": sum(map(len, false.values())),
                "overlapping_same_species": overlapping,
                "fold_sizes": [len(valid) for _, valid in splits],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("eda", "cv", "select", "export"))
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--run-id")
    parser.add_argument("--plan", type=Path, default=COMP / "selection-plan.json")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action == "eda":
        eda()
        return
    plan_file = args.plan.resolve()
    raw = json.loads(plan_file.read_text(encoding="utf-8-sig"))
    plan = SelectionPlan(tuple(raw["candidates"]), float(raw["tolerance"]))
    if args.action == "cv":
        if not args.trial or not args.run_id or args.run_id not in plan.candidates:
            parser.error("cv requires --trial and a --run-id registered in --plan")
        run(
            args.trial,
            args.run_id,
            plan_file,
            plan,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        return
    if args.receipt is None:
        parser.error("--receipt is required")
    store = RunStore(COMP)
    *_, labels, splits, evaluation = load_cv()
    assess = dense_oof(
        labels, splits, lwlrap, expected_shape=labels.shape, data_hash=evaluation.data_hash
    )
    if args.action == "select":
        selection = select_runs(store, plan, assess)
        write_selection(args.receipt, selection)
        print(selection)
        return
    if not args.run_id:
        parser.error("export requires --run-id")
    choice = verify_selection(store, plan, args.receipt, assess)
    chosen = store.record(choice.winner)
    query = test_paths()
    ids = [item.stem for item in query]
    if chosen.params["test_hash"] != files_sha256(query):
        raise ValueError("テスト音声が CV 実行時から変わっています")

    def write(output: Path, _winner: object) -> None:
        saved_ids = json.loads(
            store.artifact_path(choice.winner, "test_ids.json").read_text(encoding="utf-8")
        )
        if saved_ids != ids:
            raise ValueError("保存テスト予測の ID が一致しません")
        scores = store.load_array(choice.winner, "test.npy")
        write_submission(output / "submission.csv", ids, scores)

    def validate(output: Path) -> None:
        validate_submission(output / "submission.csv", ids)

    print(
        publish_export(
            store,
            args.run_id,
            plan=plan,
            receipt=args.receipt,
            assess=assess,
            source=source(plan_file, str(chosen.params["trial"])),
            write=write,
            required_outputs=("submission.csv",),
            validate_outputs=validate,
        )
    )


if __name__ == "__main__":
    main()
