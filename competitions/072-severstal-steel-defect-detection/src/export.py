"""全学習画像で再学習し、非公開 Kaggle 提出用の重みと Notebook を作る。"""

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch
from model import SteelUNet
from steel import SHAPE, load_labels
from train import ROOT, WEIGHTS, SteelImages, fit, fit_presence

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256, peak_memory_mb
from kumite.runs import RunStore

SLUG = "severstal-steel-defect-detection"
INFERENCE = """
from PIL import Image

test_dir, = (path for path in Path('/kaggle/input').rglob('test_images') if path.is_dir())
checkpoint, = Path('/kaggle/input').rglob('steel-checkpoint.pt')
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = SteelUNet()
model.load_state_dict(state['weights'])
model.eval()
gate = None
if state['variant'] == 'gate':
    gate = make_presence_model()
    gate.load_state_dict(state['gate'])
    gate.eval()
torch.set_num_threads(2)
height, width = state['shape']
images = sorted(test_dir.glob('*.jpg'))
if not images:
    raise ValueError('No test JPEGs found')
with torch.inference_mode(), Path('/kaggle/working/submission.csv').open(
    'w', newline='', encoding='utf-8'
) as stream:
    writer = csv.writer(stream)
    writer.writerow(('ImageId_ClassId', 'EncodedPixels'))
    for path in images:
        with Image.open(path) as image:
            pixels = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
            x = normalize(torch.from_numpy(np.array(pixels)).permute(2, 0, 1)[None], torch.device('cpu'))
        probability = F.interpolate(
            model(x).sigmoid(), size=SHAPE, mode='bilinear', align_corners=False
        )
        if tuple(probability.shape) != (1, 4, *SHAPE) or not torch.isfinite(probability).all():
            raise ValueError(f'Unexpected prediction for {path.name}')
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError(f'Invalid probability for {path.name}')
        masks = probability[0].numpy() >= 0.5
        if gate is not None:
            presence = gate(x).sigmoid()
            if tuple(presence.shape) != (1, 4) or not torch.isfinite(presence).all():
                raise ValueError(f'Unexpected presence prediction for {path.name}')
            masks &= presence[0, :, None, None].numpy() >= 0.5
        for class_id, mask in zip(CLASSES, masks, strict=True):
            encoded = '' if state['variant'] == 'area' and mask.sum() < 32 else encode_rle(mask)
            writer.writerow((f'{path.name}_{class_id}', encoded))
"""


def write_bundle(
    output: Path,
    model: torch.nn.Module,
    shape: tuple[int, int],
    variant: str,
    owner: str,
    run_id: str,
    gate: torch.nn.Module | None = None,
) -> None:
    name = f"kumite-steel-{run_id.lower()}"
    dataset_id = f"{owner}/{name}-weights"
    torch.save(
        {
            "weights": {key: value.cpu() for key, value in model.state_dict().items()},
            "shape": shape,
            "variant": variant,
            "gate": {key: value.cpu() for key, value in gate.state_dict().items()}
            if gate
            else None,
        },
        output / "steel-checkpoint.pt",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps({"id": dataset_id, "title": f"{name} weights", "licenses": [{"name": "other"}]}),
        encoding="utf-8",
    )
    source_dir = Path(__file__).parent
    source = "\n\n".join(
        [(source_dir / name).read_text(encoding="utf-8") for name in ("steel.py", "model.py")]
        + [INFERENCE]
    )
    write_cpu_notebook(output, source, kernel_id=f"{owner}/{name}", title=name, competition=SLUG)
    metadata = output / "kernel-metadata.json"
    values = json.loads(metadata.read_text(encoding="utf-8"))
    values["dataset_sources"] = [dataset_id]
    metadata.write_text(json.dumps(values, indent=2), encoding="utf-8")


def stage_upload(store: RunStore, run_id: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, files in (
        ("dataset", ("dataset-metadata.json", "steel-checkpoint.pt")),
        ("kernel", ("kernel-metadata.json", "submit.ipynb")),
    ):
        directory = output / name
        directory.mkdir()
        for filename in files:
            shutil.copyfile(store.artifact_path(run_id, filename), directory / filename)


def export(
    owner: str, run_id: str, source: str, selection: Path | None, full_baseline: str | None
) -> None:
    store = RunStore(ROOT)
    cv = store.record(source)
    variant = str(cv.params["variant"])
    inputs = store.input_hashes(source, "oof.csv")
    if selection is None:
        if variant != "baseline":
            raise ValueError("最終モデルの再学習には --selection が必要です")
    else:
        evidence = json.loads(selection.read_text(encoding="utf-8"))
        if (
            evidence["selected_run"] != source
            or evidence["oof_hashes"][source] != cv.artifacts["oof.csv"]
        ):
            raise ValueError("選定記録と CV run が一致しません")
        inputs["selection.json"] = file_sha256(selection)
    shape = SHAPE if variant == "full" else (128, 800)
    if (full_baseline is not None) != (variant in {"area", "gate"}):
        raise ValueError("area と gate の全件推論には --full-baseline が必要です")
    checkpoint = None
    if full_baseline is not None:
        full = store.record(full_baseline)
        baseline_source = full.params.get("source_run")
        if (
            full.params.get("variant") != "baseline"
            or full.params.get("score_source") != "source_cv"
            or full.data_hash != cv.data_hash
            or full.split_hash != cv.split_hash
            or f"{baseline_source}/fold-0.pt" not in cv.inputs
        ):
            raise ValueError("全件学習済みの基準 run が CV の基準と一致しません")
        checkpoint = store.artifact_path(full_baseline, "steel-checkpoint.pt")
        inputs.update(store.input_hashes(full_baseline, "steel-checkpoint.pt"))
    data = ROOT / "data"
    labels = load_labels(data)
    dataset = SteelImages(data / "train_images", labels, list(labels), shape, resize_masks=True)
    with store.reserve(run_id) as reservation:
        started = time.monotonic()
        device = torch.device("cuda")
        version = code_version(REPO_ROOT)
        if checkpoint is None:
            model = fit(dataset, data / WEIGHTS, variant, device)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if tuple(state["shape"]) != shape or state["variant"] != "baseline":
                raise ValueError("全件学習済み重みの設定が一致しません")
            model = SteelUNet()
            model.load_state_dict(state["weights"])
        gate = None
        if variant == "gate":
            presence_data = SteelImages(
                data / "train_images",
                labels,
                list(labels),
                shape,
                resize_masks=False,
                presence=True,
            )
            gate = fit_presence(presence_data, data / WEIGHTS, device)
        record = replace(
            cv,
            run_id=run_id,
            created_at="",
            late_score=None,
            conclusion="",
            next_experiment="",
            params={**cv.params, "score_source": "source_cv", "source_run": source},
            inputs=inputs,
            runtime_sec=time.monotonic() - started,
            peak_memory_mb=peak_memory_mb(),
            code_version=version,
            artifacts={},
        )
        final = store.publish(
            reservation,
            record,
            lambda output: write_bundle(output, model, shape, variant, owner, run_id, gate),
        )
        print(final, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("source")
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--full-baseline", help="area と gate で再利用する全件学習済みの基準 run ID"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("出力先には新規ディレクトリを指定してください")
    export(args.owner, args.run_id, args.source, args.selection, args.full_baseline)
    stage_upload(RunStore(ROOT), args.run_id, args.output)


if __name__ == "__main__":
    main()
