"""既存の分割・OOFだけによる事後診断。モデル選定・学習・提出は行わない。"""

import argparse
import json
from pathlib import Path

import numpy as np
from dataset import CLASSES, OUTPUT, ROOT, load_prepared, make_folds
from split_diagnostic import fold_profiles

from kumite.checks import validate_probabilities
from kumite.experiments import compare_runs
from kumite.provenance import array_sha256, file_sha256
from kumite.runs import RunStore


def oof_decomposition(
    y: np.ndarray, probabilities: dict[str, np.ndarray], silence_share: float
) -> dict:
    """構成比を変える感度分析。条件付きの正答率はCVのまま固定する。"""
    if not 0 <= silence_share <= 1 or len(y) == 0 or not np.isin(y, np.arange(12)).all():
        raise ValueError("ラベルまたは仮定するsilence比率が不正です")
    silence = y == 10
    if not silence.any() or silence.all():
        raise ValueError("silenceと非silenceの両方が必要です")
    result = {}
    for run, values in probabilities.items():
        validate_probabilities(values, len(y), 12)
        predicted = values.argmax(1)
        correct = predicted == y
        by_class = {}
        for i, name in enumerate(CLASSES):
            mask = y == i
            n = int(mask.sum())
            by_class[name] = dict(
                n=n,
                share=n / len(y),
                correct=int(correct[mask].sum()),
                recall=float(correct[mask].mean()) if n else None,
                accuracy_contribution=float(correct[mask].sum() / len(y)),
            )
        groups = {}
        for name, mask in (("silence", silence), ("non_silence", ~silence)):
            groups[name] = dict(
                n=int(mask.sum()),
                multiclass_accuracy=float(correct[mask].mean()),
                binary_detection_recall=float(((predicted == 10) == silence)[mask].mean()),
            )
        sensitivity = []
        for proportion in sorted({float(silence.mean()), 0.05, silence_share, 0.15, 0.20}):
            weighted = (
                proportion * groups["silence"]["multiclass_accuracy"]
                + (1 - proportion) * groups["non_silence"]["multiclass_accuracy"]
            )
            sensitivity.append(dict(silence_share=proportion, weighted_accuracy=weighted))
        result[run] = dict(
            oof_accuracy=float(correct.mean()),
            classes=by_class,
            groups=groups,
            sensitivity=sensitivity,
        )
    return result


def main(output: Path) -> None:
    frame, _, waves, _, summary = load_prepared(deep=True)
    store = RunStore(ROOT)
    rms = np.concatenate(
        [
            np.sqrt(np.mean(np.asarray(waves[start : start + 512]) ** 2, axis=1, dtype=np.float64))
            for start in range(0, len(waves), 512)
        ]
    )
    profiles, comparisons = {}, {}
    external = json.loads((OUTPUT / "external_evaluation.json").read_text(encoding="utf-8"))
    proportion = external["P0"]["private"]
    for seed, base, candidate in ((42, "E01", "E05-r2"), (43, "E01-s43", "E05-s43")):
        folds = make_folds(frame, seed)
        records = [store.record(run) for run in (base, candidate)]
        compare_runs(*records)
        for record in records:
            if record.data_hash != summary["data_hash"] or record.split_hash != array_sha256(folds):
                raise ValueError("診断対象のデータまたは分割が一致しません")
        profiles[str(seed)] = fold_profiles(frame, folds, rms)
        comparisons[str(seed)] = oof_decomposition(
            frame.label.to_numpy(dtype=np.int64),
            {run: store.load_array(run, "oof.npy") for run in (base, candidate)},
            proportion,
        )
    result = dict(
        post_hoc_only=True,
        assumption="クラス条件付き正答率と非silence内の相対クラス比率をCVのまま固定。Private精度の推定やモデル選定には使わない。",
        data_hash=summary["data_hash"],
        private_silence_share_from_constant_submission=proportion,
        external_evaluation_sha256=file_sha256(OUTPUT / "external_evaluation.json"),
        folds=profiles,
        oof=comparisons,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "review" / "diagnostics.json")
    main(parser.parse_args().output)
