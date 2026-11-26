"""固定分割の学習、保存 OOF の再採点、候補選定。"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from data import ROOT, SEED, Pair, load_pairs, make_folds
from model import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL,
    STRIDE,
    decode_pair,
    encode,
    fit,
    predict,
    tokenizer,
)
from spans import Span, micro_f1

from kumite.experiments import ExperimentRecord
from kumite.provenance import array_sha256, code_version, files_sha256
from kumite.quality_gates import (
    SelectionPlan,
    SourceSpec,
    publish_gated_cv,
    reserve_cv,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore

REPO = ROOT.parents[1]
DATA = ROOT / "data"
PROBABILITIES = "token-scores.json"
PREDICTIONS = "oof.json"
TRIALS = {
    "B00": "項目説明を条件とするベースライン",
    "E01": "項目説明を項目 ID に変更",
    "E02": "陽性ラベルを文字の半数以上が正解と重なるトークンに限定",
    "E03": "両側が陽性の空白を連結",
    "E04": "期間項目309の予測から2週または年を示す区間を除外",
}


def plan() -> SelectionPlan:
    payload = json.loads((ROOT / "selection-plan.json").read_text(encoding="utf-8-sig"))
    return SelectionPlan(tuple(payload["candidates"]), payload["tolerance"])


def source(trial: str) -> SourceSpec:
    prefix = ROOT.relative_to(REPO).as_posix()
    files = tuple(
        f"{prefix}/src/{name}.py"
        for name in ("train", "model", "data", "spans", "export", "notebook")
    )
    shared = (
        "__init__",
        "crossfit",
        "experiments",
        "paths",
        "provenance",
        "quality_gates",
        "runs",
        "source_audit",
    )
    return SourceSpec(
        REPO,
        (
            *files,
            *(f"src/kumite/{name}.py" for name in shared),
            "pyproject.toml",
            "uv.lock",
            f"{prefix}/quality-gates.toml",
            f"{prefix}/selection-plan.json",
        ),
        {
            "trial": trial,
            "model": MODEL,
            "seed": SEED,
            "epochs": EPOCHS,
            "length": MAX_LENGTH,
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "threshold": 0.5,
        },
        ("src", f"{prefix}/src"),
    )


def dataset() -> tuple[list[Pair], list[int], str, str]:
    pairs = load_pairs(DATA, training=True)
    folds = make_folds(pairs)
    data_hash = files_sha256(
        DATA / name for name in ("train.csv", "features.csv", "patient_notes.csv")
    )
    return pairs, folds, data_hash, array_sha256(np.asarray(folds, dtype=np.int8))


def score(
    pairs: list[Pair],
    folds: list[int],
    predictions: list[list[Span]],
) -> tuple[list[float], float]:
    rows = [
        (len(pair.text), pair.spans, output)
        for pair, output in zip(pairs, predictions, strict=True)
    ]
    return (
        [
            micro_f1(row for row, group in zip(rows, folds, strict=True) if group == fold)
            for fold in range(5)
        ],
        micro_f1(rows),
    )


def assess(artifact: Callable[[str], Path], record: ExperimentRecord) -> tuple[list[float], float]:
    pairs, folds, data_hash, split_hash = dataset()
    if (record.data_hash, record.split_hash) != (data_hash, split_hash):
        msg = "OOF のデータまたは分割ハッシュが一致しません"
        raise ValueError(msg)
    rows = json.loads(artifact(PREDICTIONS).read_text(encoding="utf-8"))
    predictions: list[list[Span]] = []
    for pair, fold, row in zip(pairs, folds, rows, strict=True):
        if row["id"] != pair.id or row["fold"] != fold:
            msg = "OOF の ID または fold が一致しません"
            raise ValueError(msg)
        spans: list[Span] = []
        for start, end in row["spans"]:
            if type(start) is not int or type(end) is not int:
                msg = "OOF の文字位置は整数が必要です"
                raise ValueError(msg)
            spans.append((start, end))
        predictions.append(spans)
    return score(pairs, folds, predictions)


def token_scores(
    pairs: list[Pair],
    folds: list[int],
    trial: str,
) -> list[dict[Span, float]]:
    tok = tokenizer()
    pad = tok.pad_token_id
    if not isinstance(pad, int):
        msg = "パディングトークン ID がありません"
        raise ValueError(msg)
    device = torch.device("cuda")
    result: list[dict[Span, float]] = [{} for _ in pairs]
    for fold in range(5):
        training = [pair for pair, group in zip(pairs, folds, strict=True) if group != fold]
        valid = [index for index, group in enumerate(folds) if group == fold]
        queries = [pairs[index] for index in valid]
        model = fit(encode(training, tok, trial), pad, device)
        scores = predict(model, encode(queries, tok, trial), len(valid), pad, device)
        for index, row in zip(valid, scores, strict=True):
            result[index] = row
        print(f"fold {fold} 完了", flush=True)
    return result


def run(trial: str, run_id: str, baseline: str) -> Path:
    store = RunStore(ROOT)
    inputs = store.input_hashes(baseline, PROBABILITIES) if trial in {"E03", "E04"} else {}
    spec = source(trial)
    with reserve_cv(store, run_id, spec, inputs, plan()) as (reservation, start):
        started = perf_counter()
        pairs, folds, data_hash, split_hash = dataset()
        if trial in {"E03", "E04"}:
            saved = json.loads(
                store.artifact_path(baseline, PROBABILITIES).read_text(encoding="utf-8")
            )
            probabilities = []
            for pair, row in zip(pairs, saved, strict=True):
                if row["id"] != pair.id:
                    msg = "依存する OOF の ID が一致しません"
                    raise ValueError(msg)
                probabilities.append({(a, b): p for a, b, p in row["scores"]})
        else:
            probabilities = token_scores(pairs, folds, trial)
        outputs = [
            decode_pair(pair, row, trial) for pair, row in zip(pairs, probabilities, strict=True)
        ]
        fold_scores, oof_score = score(pairs, folds, outputs)
        record = ExperimentRecord(
            competition_id="055",
            run_id=run_id,
            metric="character micro-F1",
            greater_is_better=True,
            data_version="official-v1",
            data_hash=data_hash,
            split_version="note-group-v1-seed42",
            split_hash=split_hash,
            seed=SEED,
            hypothesis=TRIALS[trial],
            changed_one_thing=TRIALS[trial],
            model=MODEL,
            params=dict(spec.settings),
            fold_scores=fold_scores,
            oof_score=oof_score,
            runtime_sec=perf_counter() - started,
            code_version=code_version(REPO),
            inputs=inputs,
        )

        def write(path: Path) -> None:
            rows = [
                {"id": pair.id, "fold": fold, "spans": spans}
                for pair, fold, spans in zip(pairs, folds, outputs, strict=True)
            ]
            (path / PREDICTIONS).write_text(json.dumps(rows), encoding="utf-8")
            if trial == "B00":
                scores = [
                    {"id": pair.id, "scores": [(a, b, p) for (a, b), p in row.items()]}
                    for pair, row in zip(pairs, probabilities, strict=True)
                ]
                (path / PROBABILITIES).write_text(
                    json.dumps(scores, allow_nan=False), encoding="utf-8"
                )

        return publish_gated_cv(
            store, reservation, record, write, source=spec, before_training=start, assess=assess
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=(*TRIALS, "select"))
    parser.add_argument("--run-id")
    parser.add_argument("--baseline", default="B00")
    args = parser.parse_args()
    if args.trial == "select":
        selection = select_runs(RunStore(ROOT), plan(), assess)
        write_selection(ROOT / "selection.json", selection)
        print(selection)
    else:
        print(run(args.trial, args.run_id or args.trial, args.baseline))
