"""Baseline または全データ再学習の非公開 CPU Notebook を生成する。"""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from inspect import getsource
from pathlib import Path

import pandas as pd
import torch
from analyze import verify_selection
from dataset import RATE, SEED, WINDOW, data_fingerprint, read_clips, read_wave, targets, windows
from dummy import SLUG
from model import Tagger, fit, predict
from train import ROOT

from kumite.audio import Features, SmallCNN
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.runs import RunStore

PUBLIC_TEST_COUNT = 3361
TOTAL_TEST_COUNT = 4481

IMPORTS = """import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile
import torch
from torch import nn

"""


def evaluation_ids(
    public_ids: list[str], template_ids: list[str], archive_ids: list[str], private_count: int
) -> list[str]:
    public = set(public_ids)
    mounted = set(archive_ids)
    if len(mounted) != len(archive_ids):
        raise ValueError("Duplicate test clip IDs in archive")
    if template_ids != public_ids:
        raise ValueError("Unexpected public template IDs or order")
    missing = public - mounted
    extra = mounted - public
    if not missing and not extra:
        return public_ids
    print(
        f"Test archive mismatch: {len(mounted)} WAVs, {len(missing)} missing public, {len(extra)} extra"
    )
    if len(extra) == private_count and (not missing or missing == public):
        return sorted(mounted)
    raise ValueError("Unexpected test archive manifest")


INFERENCE = """
torch.set_num_threads(4)
sample, = Path('/kaggle/input').rglob('sample_submission.csv')
weights, = Path('/kaggle/input').rglob('freesound-fold0.pt')
template = pd.read_csv(sample, dtype={'fname': str})
if list(template.columns) != ['fname', *TAGS]:
    raise ValueError('Unexpected submission columns')
audio = ZipFile(sample.with_name('test.zip'))
members = [name for name in audio.namelist() if name.endswith('.wav')]
files = {Path(name).name: name for name in members}
names = evaluation_ids(PUBLIC_IDS, template['fname'].tolist(), [Path(name).name for name in members], PRIVATE_COUNT)
device = torch.device('cpu')
models = []
for fold, expected in enumerate(WEIGHTS):
    model = Tagger(len(TAGS)).eval()
    checkpoint = weights.with_name(f'freesound-fold{fold}.pt')
    with checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('Checkpoint hash mismatch')
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    models.append(model)
rows = []
with audio:
    for name in names:
        wave = read_wave(BytesIO(audio.read(files[name])))
        rows.append(np.mean([predict(model, wave, device, maximum=MAXIMUM) for model in models], axis=0))
scores = np.stack(rows)
if scores.shape != (len(names), len(TAGS)) or not np.isfinite(scores).all():
    raise ValueError('Invalid prediction shape or non-finite scores')
if np.any((scores < 0) | (scores > 1)):
    raise ValueError('Invalid score range')
submission = pd.DataFrame(scores, columns=TAGS)
submission.insert(0, 'fname', names)
submission.to_csv('/kaggle/working/submission.csv', index=False)
"""


def notebook_source(
    tags: list[str], weights: list[str], maximum: bool, public_ids: list[str], private_count: int
) -> str:
    definitions = "\n\n".join(
        getsource(item)
        for item in (read_wave, windows, Features, SmallCNN, Tagger, predict, evaluation_ids)
    )
    config = f"RATE = {RATE}\nWINDOW = {WINDOW}\nTAGS = {tags!r}\nWEIGHTS = {weights!r}\nMAXIMUM = {maximum!r}\n"
    config += f"PUBLIC_IDS = {public_ids!r}\nPRIVATE_COUNT = {private_count}\n"
    source = IMPORTS + config + "\n\n" + definitions + "\n" + INFERENCE
    compile(source, "freesound-notebook", "exec")
    return source


def refit(store: RunStore, run_id: str, selected_run: str, selection: Path, seconds: float) -> Path:
    start = time.monotonic()
    with store.reserve(run_id) as reservation:
        source_bytes = selection.read_bytes()
        choice = json.loads(source_bytes)
        source = store.record(selected_run)
        experiment = source.params["experiment"]
        if experiment not in {"E01", "E02", "E03", "E04", "E05"} or source.params.get("benchmark"):
            raise ValueError("Unsupported final experiment")
        verify_selection(store, choice, selected_run)
        root = ROOT / "data"
        curated = read_clips(root, "curated")
        noisy = read_clips(root, "noisy")
        if data_fingerprint(root, curated + noisy) != source.data_hash:
            raise ValueError("Final training data differs from the selected run")
        tags = source.params["tags"]
        allowed = noisy if experiment in {"E02", "E05"} else []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fit(
            [read_wave(clip.path) for clip in curated],
            targets(curated, tags),
            [clip.path for clip in allowed],
            targets(allowed, tags),
            updates=source.params["updates"],
            seed=SEED,
            device=device,
            deadline=start + seconds,
            time_mask=experiment == "E04",
            staged_noisy=experiment == "E05",
        )
        record = replace(
            source,
            run_id=run_id,
            created_at="",
            split_version="full-data-refit-v1",
            hypothesis="CV で固定した構成の外部評価",
            changed_one_thing="全学習データで再学習",
            params={**source.params, "source_run": selected_run, "score_source": "source_cv"},
            inputs={
                **store.input_hashes(selected_run, "oof.npy", "labels.npy"),
                "selection.json": hashlib.sha256(source_bytes).hexdigest(),
            },
            artifacts={},
            runtime_sec=time.monotonic() - start,
            peak_memory_mb=None,
            late_score=None,
            conclusion="",
            next_experiment="",
            code_version=code_version(REPO_ROOT),
        )

        def write(path: Path) -> None:
            (path / "selection.json").write_bytes(source_bytes)
            torch.save(
                {name: value.cpu() for name, value in model.state_dict().items()}, path / "final.pt"
            )

        return store.publish(reservation, record, write)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--refit-run")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    store = RunStore(ROOT)
    record = store.record(args.run_id)
    if args.output.exists():
        raise FileExistsError(args.output)
    public = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype={"fname": str})["fname"]
    if len(public) != PUBLIC_TEST_COUNT or public.isna().any() or public.duplicated().any():
        raise ValueError("Invalid official public manifest")
    public_ids = public.tolist()
    if args.selection is None:
        if record.params.get("experiment") not in {"E01", "E05"} or record.params.get("benchmark"):
            raise ValueError("A complete E01 or E05 CV run is required")
        checkpoint_paths = [store.artifact_path(args.run_id, f"fold{i}.pt") for i in range(3)]
        suffix = "baseline" if record.params["experiment"] == "E01" else "e05-fold-diagnostic"
    else:
        if not args.refit_run or args.seconds is None or args.seconds <= 0:
            parser.error("--selection requires --refit-run and a positive --seconds")
        refit(store, args.refit_run, args.run_id, args.selection, args.seconds)
        checkpoint_paths = [store.artifact_path(args.refit_run, "final.pt")]
        suffix = "final"
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / "weights"
    weights.mkdir()
    for index, checkpoint in enumerate(checkpoint_paths):
        shutil.copyfile(checkpoint, weights / f"freesound-fold{index}.pt")
    (weights / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"Kumite Freesound {suffix} weights",
                "id": args.dataset,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    kernel = args.output / "kernel"
    write_cpu_notebook(
        kernel,
        notebook_source(
            record.params["tags"],
            [file_sha256(path) for path in checkpoint_paths],
            record.params["experiment"] == "E03",
            public_ids,
            TOTAL_TEST_COUNT - PUBLIC_TEST_COUNT,
        ),
        kernel_id=f"{args.owner}/kumite-freesound-{suffix}",
        title=f"Kumite Freesound {suffix}",
        competition=SLUG,
    )
    metadata_path = kernel / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_sources"] = [args.dataset]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
