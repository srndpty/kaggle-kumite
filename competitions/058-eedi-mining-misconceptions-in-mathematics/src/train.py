"""Eedi の固定分割・実験実行・OOF 選定。"""

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from ranking import (
    E5_REVISION,
    KEY,
    LABEL,
    load_mapping,
    read_queries,
    rerank,
    retrieve,
    submission,
    train_ranker,
    validate_ranking,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import GroupKFold

from kumite.bootstrap import group_bootstrap_diff
from kumite.checks import check_label_independence, shift_labels
from kumite.crossfit import CrossFitResult
from kumite.cv_run import Evaluation, Trial, publish_cv_run
from kumite.experiments import ExperimentRecord
from kumite.kaggle_notebook import write_cpu_notebook
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256, files_sha256, peak_memory_mb
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "outputs" / "e5"
SLUG = "eedi-mining-misconceptions-in-mathematics"
STORE = RunStore(ROOT)
TRIALS = {
    "E00": "文字 TF-IDF を基準にする",
    "E01": "意味検索で候補漏れを減らす",
    "E02": "文字検索との融合で候補漏れを減らす",
    "E03": "候補集合を固定して順位を改善する",
    "E04": "構成概念の説明を追加して検索する",
}


def splits_for(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(
        GroupKFold(n_splits=5, shuffle=True, random_state=42).split(frame, groups=frame["group"])
    )


def reciprocal(y: np.ndarray, pred: np.ndarray, cutoff: int = 25) -> np.ndarray:
    """単一正解の逆順位。候補IDの一意性は採点前に validate_ranking で検査する。"""
    matches = pred[:, :cutoff] == y[:, None]
    return np.max(matches / np.arange(1, matches.shape[1] + 1), axis=1)


def map25(y: np.ndarray, pred: np.ndarray) -> float:
    return float(reciprocal(y, pred).mean())


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    hit100 = (pred == y[:, None]).any(axis=1)
    return {
        "n": len(y),
        "map25": map25(y, pred) if len(y) else None,
        "recall25": float((reciprocal(y, pred) > 0).mean()) if len(y) else None,
        "recall100": float(hit100.mean()) if len(y) else None,
        "conditional_map25": map25(y[hit100], pred[hit100]) if hit100.any() else None,
    }


def evaluate(oof: np.ndarray, y: np.ndarray, known: np.ndarray) -> dict[str, object]:
    return {
        "all": metrics(y, oof),
        "known": metrics(y[known], oof[known]),
        "unknown": metrics(y[~known], oof[~known]),
    }


def fit_predict(
    train: np.ndarray,
    valid: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    model = train_ranker(candidates[train], features[train], y[train])
    return rerank(model, candidates[valid], features[valid])


def has_shared_word(query: str, label: str) -> bool:
    excluded = ENGLISH_STOP_WORDS | {"answer", "correct", "incorrect"}
    left = set(re.findall(r"[a-z]+", query.lower())) - excluded
    right = set(re.findall(r"[a-z]+", label.lower())) - excluded
    return bool(left & right)


def load_cv() -> tuple[pd.DataFrame, Evaluation]:
    frame = read_queries(DATA / "train.csv", labeled=True)
    folds = np.empty(len(frame), dtype=np.int8)
    for number, (_, valid) in enumerate(splits_for(frame)):
        folds[valid] = number
    evaluation = Evaluation(
        "058",
        "MAP@25",
        True,
        map25,
        files_sha256(
            DATA / name
            for name in (
                "misconception_mapping.csv",
                "sample_submission.csv",
                "test.csv",
                "train.csv",
            )
        ),
        "v1",
        sha256(folds.tobytes()).hexdigest(),
        42,
    )
    return frame, evaluation


def cv(trial: str, run_id: str) -> None:
    with STORE.reserve(run_id) as reservation:
        start = perf_counter()
        frame, evaluation = load_cv()
        ids, names = load_mapping(DATA / "misconception_mapping.csv")
        y = frame[LABEL].to_numpy(dtype=int)
        if not np.isin(y, ids).all():
            raise ValueError("学習ラベルがマッピングにありません")
        ranking, features = retrieve(frame, names, trial, MODEL)
        candidates = ids[ranking]
        validate_ranking(candidates, len(y), ids, 100)
        splits = splits_for(frame)
        oof = candidates.copy()
        known = np.empty(len(y), dtype=bool)
        folds = np.empty(len(y), dtype=int)
        for number, (train, valid) in enumerate(splits):
            known[valid] = np.isin(y[valid], y[train])
            folds[valid] = number
            folds_for_validation = np.zeros(len(y), dtype=bool)
            folds_for_validation[valid] = True
            if trial == "E03":
                changed = shift_labels(y, folds_for_validation, int(ids.max()) + 1, seed=42)

                def build(
                    labels: np.ndarray, tr: np.ndarray = train, va: np.ndarray = valid
                ) -> list[np.ndarray]:
                    return [fit_predict(tr, va, labels, candidates, features)]

                check_label_independence(build, y, changed)
                oof[valid] = build(y)[0]
        validate_ranking(oof, len(y), ids, 100)
        if not np.array_equal(np.sort(candidates, axis=1), np.sort(oof, axis=1)):
            raise ValueError("候補集合が変化しました")
        summary = evaluate(oof, y, known)
        print(json.dumps(summary))

        def write(out: Path) -> None:
            np.save(out / "candidates.npy", candidates)
            pd.DataFrame({KEY: frame[KEY], "fold": folds, "known": known}).to_csv(
                out / "folds.csv", index=False
            )
            (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        params = {
            "trial": trial,
            "source_hash": file_sha256(Path(__file__).with_name("ranking.py")),
        }
        if trial != "E00":
            params.update(
                e5_revision=E5_REVISION,
                e5_hash=files_sha256(sorted(path for path in MODEL.iterdir() if path.is_file())),
            )
        publish_cv_run(
            STORE,
            reservation,
            evaluation,
            Trial(TRIALS[trial], TRIALS[trial], trial, "official-v1", params),
            y,
            splits,
            CrossFitResult(oof, np.empty(0), [{} for _ in splits]),
            perf_counter() - start,
            write,
            save_query=False,
        )


def cv_trial(record: ExperimentRecord, evaluation: Evaluation) -> str:
    if (
        record.competition_id != evaluation.competition_id
        or record.metric != evaluation.metric
        or record.greater_is_better is not evaluation.greater_is_better
        or record.data_hash != evaluation.data_hash
        or record.split_hash != evaluation.split_hash
        or record.split_version != evaluation.split_version
        or record.seed != evaluation.seed
        or "source_cv_run" in record.params
    ):
        raise ValueError(f"{record.run_id} の CV 評価条件が一致しません")
    trial = record.params.get("trial")
    if not isinstance(trial, str) or trial not in TRIALS:
        raise ValueError(f"{record.run_id} の trial が不正です")
    if trial != "E00" and (
        record.params.get("e5_revision") != E5_REVISION or not record.params.get("e5_hash")
    ):
        raise ValueError(f"{record.run_id} の E5 リビジョンまたはハッシュが不正です")
    return trial


def build_selection(
    run_ids: list[str],
    baseline: str,
    frame: pd.DataFrame,
    evaluation: Evaluation,
    *,
    require_all_trials: bool = False,
) -> dict[str, object]:
    if len(set(run_ids)) != len(run_ids) or baseline not in run_ids:
        raise ValueError("E00 を含む異なる run ID を指定してください")
    y = frame[LABEL].to_numpy(dtype=int)
    ids, names = load_mapping(DATA / "misconception_mapping.csv")
    labels_by_id = dict(zip(ids, names, strict=True))
    overlap = np.array(
        [has_shared_word(q, labels_by_id[t]) for q, t in zip(frame["text"], y, strict=True)]
    )
    comparisons = []
    fusion_candidates: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    trials: set[str] = set()
    e5_hashes: set[str] = set()
    for run_id in run_ids:
        record = STORE.record(run_id)
        trial = cv_trial(record, evaluation)
        if trial in trials:
            raise ValueError(f"trial {trial} が重複しています")
        trials.add(trial)
        if run_id == baseline and trial != "E00":
            raise ValueError("ベースラインには E00 の run を指定してください")
        if trial != "E00":
            e5_hashes.add(str(record.params["e5_hash"]))
            if len(e5_hashes) > 1:
                raise ValueError("比較候補の E5 ファイルハッシュが一致しません")
        path = STORE.runs_dir / run_id / "oof.npy"
        oof_hash = file_sha256(path)
        if oof_hash != record.artifacts.get("oof.npy"):
            raise ValueError(f"{run_id} の OOF ハッシュが一致しません")
        pred = np.load(path, allow_pickle=False)
        validate_ranking(pred, len(y), ids, 100)
        predictions[run_id] = pred
        if trial in {"E02", "E03"}:
            candidate_file = path.with_name("candidates.npy")
            if file_sha256(candidate_file) != record.artifacts.get("candidates.npy"):
                raise ValueError(f"{run_id} の候補ハッシュが一致しません")
            candidates = np.load(candidate_file, allow_pickle=False)
            validate_ranking(candidates, len(y), ids, 100)
            if not np.array_equal(np.sort(candidates, axis=1), np.sort(pred, axis=1)):
                raise ValueError(f"{run_id} の再ランキングで候補が変わりました")
            fusion_candidates[trial] = candidates
        score = map25(y, pred)
        if not np.isclose(score, record.oof_score, rtol=0, atol=1e-12):
            raise ValueError(f"{run_id} の記録と OOF の採点が一致しません")
        hit100 = (pred == y[:, None]).any(axis=1)
        comparisons.append(
            {
                "run": run_id,
                "score": score,
                "oof_sha256": oof_hash,
                "shared_word_recall100": float(hit100[overlap].mean()) if overlap.any() else None,
                "no_shared_word_recall100": float(hit100[~overlap].mean())
                if (~overlap).any()
                else None,
            }
        )
    if len(fusion_candidates) == 2 and not np.array_equal(
        fusion_candidates["E02"], fusion_candidates["E03"]
    ):
        raise ValueError("E02 と E03 の候補が一致しません")
    if require_all_trials and trials != TRIALS.keys():
        raise ValueError("最終提出の選定には E00〜E04 の全 trial が必要です")
    best = min(comparisons, key=lambda entry: (-float(entry["score"]), str(entry["run"])))
    pred = predictions[str(best["run"])]
    interval = group_bootstrap_diff(
        reciprocal(y, predictions[baseline]),
        reciprocal(y, pred),
        frame["group"].to_numpy(),
        transform=lambda values: values,
    )
    known = np.empty(len(y), dtype=bool)
    for train, valid in splits_for(frame):
        known[valid] = np.isin(y[valid], y[train])
    errors = {}
    for name, mask in (("known", known), ("unknown", ~known)):
        labels, ranks = y[mask], pred[mask]
        hit25 = reciprocal(labels, ranks) > 0
        hit100 = (ranks == labels[:, None]).any(axis=1)
        errors[name] = {
            "missing_from_100": int((~hit100).sum()),
            "rank_26_to_100": int((hit100 & ~hit25).sum()),
            "rank_2_to_25": int((hit25 & (ranks[:, 0] != labels)).sum()),
            "rank_1": int((ranks[:, 0] == labels).sum()),
        }
    return {
        "selected_run": best["run"],
        "baseline_run": baseline,
        "comparisons": comparisons,
        "data_hash": evaluation.data_hash,
        "split_hash": evaluation.split_hash,
        "delta_vs_e00_95ci": [interval.low, interval.high],
        "rows_without_shared_word": int((~overlap).sum()),
        "errors": errors,
    }


def select(run_ids: list[str], output: Path, baseline: str) -> None:
    frame, evaluation = load_cv()
    result = build_selection(run_ids, baseline, frame, evaluation)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result)


def dummy_notebook(output: Path, owner: str) -> None:
    source = """
from pathlib import Path
import pandas as pd

mapping_path, = Path('/kaggle/input').rglob('misconception_mapping.csv')
mapping = pd.read_csv(mapping_path)
sample = pd.read_csv(mapping_path.with_name('sample_submission.csv'))
sample['MisconceptionId'] = ' '.join(map(str, sorted(mapping.MisconceptionId)[:25]))
sample.to_csv('/kaggle/working/submission.csv', index=False)
"""
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/kumite-eedi-p0",
        title="Kumite Eedi P0",
        competition=SLUG,
    )


def notebook(
    trial: str,
    output: Path,
    owner: str,
    dataset: str | None,
    ranker_text: str | None,
    model_code: str,
) -> None:
    expected = {
        name: file_sha256(DATA / name) for name in ("misconception_mapping.csv", "train.csv")
    }
    weights = (
        {p.name: file_sha256(p) for p in MODEL.iterdir() if p.is_file()} if trial != "E00" else {}
    )
    source = (
        model_code
        + f"""
from hashlib import sha256


def verify_assets(folder, hashes):
    for name, expected in hashes.items():
        with (folder / name).open('rb') as stream:
            actual = sha256()
            while chunk := stream.read(1 << 20):
                actual.update(chunk)
        if actual.hexdigest() != expected:
            raise ValueError(f'Unexpected contents: {{name}}')

data, = (p.parent for p in Path('/kaggle/input').rglob('misconception_mapping.csv'))
verify_assets(data, {expected!r})
ids, names = load_mapping(data / 'misconception_mapping.csv')
sample = pd.read_csv(data / 'sample_submission.csv')
test = read_queries(data / 'test.csv', labeled=False)
trial = {trial!r}
model_dir = Path('/unused')
if trial != 'E00':
    weight, = Path('/kaggle/input').rglob('model.safetensors')
    model_dir = weight.parent
    verify_assets(model_dir, {weights!r})
candidates, features = retrieve(test, names, trial, model_dir)
pred = ids[candidates]
if trial == 'E03':
    pred = rerank(Booster(model_str={ranker_text!r}), pred, features)
submission(sample, test, pred[:, :25], ids).to_csv('/kaggle/working/submission.csv', index=False)
"""
    )
    write_cpu_notebook(
        output,
        source,
        kernel_id=f"{owner}/kumite-eedi-{trial.lower()}",
        title=f"Kumite Eedi {trial}",
        competition=SLUG,
    )
    if dataset is not None:
        metadata = output / "kernel-metadata.json"
        config = json.loads(metadata.read_text(encoding="utf-8"))
        config["dataset_sources"] = [dataset]
        metadata.write_text(json.dumps(config, indent=2), encoding="utf-8")


def export_run(
    cv_run: str, run_id: str, owner: str, dataset: str | None, selection: Path | None
) -> None:
    with STORE.reserve(run_id) as reservation:
        start = perf_counter()
        record = STORE.record(cv_run)
        frame, evaluation = load_cv()
        trial = cv_trial(record, evaluation)
        model_source = Path(__file__).with_name("ranking.py").read_bytes()
        if sha256(model_source).hexdigest() != record.params.get("source_hash"):
            raise ValueError("CV 時点の推論ソースと一致しません")
        oof_path = STORE.runs_dir / cv_run / "oof.npy"
        oof_hash = file_sha256(oof_path)
        if oof_hash != record.artifacts.get("oof.npy"):
            raise ValueError("保存済み OOF のハッシュが一致しません")
        inputs = {"cv_oof": oof_hash}
        if selection is not None:
            selection_source = selection.read_bytes()
            saved = json.loads(selection_source)
            try:
                baseline = saved["baseline_run"]
                run_ids = [item["run"] for item in saved["comparisons"]]
            except (KeyError, TypeError) as exc:
                raise ValueError("選定記録の形式が不正です") from exc
            if not isinstance(baseline, str) or not all(isinstance(run, str) for run in run_ids):
                raise ValueError("選定記録の run ID が不正です")
            rebuilt = build_selection(run_ids, baseline, frame, evaluation, require_all_trials=True)
            if saved != rebuilt or rebuilt["selected_run"] != cv_run:
                raise ValueError("CV run が選定記録と一致しません")
            inputs["selection"] = sha256(selection_source).hexdigest()
        if trial != "E00":
            if selection is None:
                raise ValueError("E00 以外の提出には --selection が必要です")
            if not dataset:
                raise ValueError("E5 を添付する --dataset が必要です")
            if (
                files_sha256(sorted(path for path in MODEL.iterdir() if path.is_file()))
                != record.params["e5_hash"]
            ):
                raise ValueError("E5 の資材が CV と一致しません")
        ids, names = load_mapping(DATA / "misconception_mapping.csv")
        test = read_queries(DATA / "test.csv", labeled=False)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        ranking, features = retrieve(test, names, trial, MODEL)
        pred = ids[ranking]
        ranker_text = None
        if trial == "E03":
            train_candidates, train_features = retrieve(frame, names, trial, MODEL)
            ranker = train_ranker(
                ids[train_candidates], train_features, frame[LABEL].to_numpy(dtype=int)
            )
            pred = rerank(ranker, pred, features)
            ranker_text = ranker.model_to_string()
        result = submission(sample, test, pred[:, :25], ids)
        final = ExperimentRecord(
            competition_id="058",
            run_id=run_id,
            metric=record.metric,
            greater_is_better=True,
            data_version=record.data_version,
            data_hash=record.data_hash,
            split_version=record.split_version,
            split_hash=record.split_hash,
            seed=record.seed,
            hypothesis=record.hypothesis,
            changed_one_thing="全学習行から提出を生成",
            model=record.model,
            params={**record.params, "source_cv_run": cv_run, "score_source": "source_cv"},
            fold_scores=record.fold_scores,
            oof_score=record.oof_score,
            runtime_sec=perf_counter() - start,
            peak_memory_mb=peak_memory_mb(),
            code_version=code_version(REPO_ROOT),
            inputs=inputs,
        )

        def write(out: Path) -> None:
            result.to_csv(out / "submission.csv", index=False)
            if ranker_text is not None:
                (out / "ranker.txt").write_text(ranker_text, encoding="utf-8")
            notebook(trial, out, owner, dataset, ranker_text, model_source.decode("utf-8-sig"))

        print(STORE.publish(reservation, final, write))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    cv_parser = modes.add_parser("cv")
    cv_parser.add_argument("trial", choices=TRIALS)
    cv_parser.add_argument("--run-id")
    select_parser = modes.add_parser("select")
    select_parser.add_argument("runs", nargs="+")
    select_parser.add_argument("--baseline", default="E00")
    select_parser.add_argument("--output", type=Path, required=True)
    p0 = modes.add_parser("p0")
    p0.add_argument("output", type=Path)
    p0.add_argument("--owner", required=True)
    final = modes.add_parser("final")
    final.add_argument("cv_run")
    final.add_argument("run_id")
    final.add_argument("--owner", required=True)
    final.add_argument("--dataset")
    final.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.mode == "cv":
        cv(args.trial, args.run_id or args.trial)
    elif args.mode == "select":
        select(args.runs, args.output, args.baseline)
    elif args.mode == "p0":
        dummy_notebook(args.output, args.owner)
    else:
        export_run(args.cv_run, args.run_id, args.owner, args.dataset, args.selection)


if __name__ == "__main__":
    main()
