"""共通 Quality Gate を使う OTTO の CV・選定・最終提出生成。"""

import argparse
import csv
import gzip
import hashlib
import json
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import tee
from pathlib import Path
from time import perf_counter
from typing import cast

import polars as pl
from candidates import METHODS, Method
from dataset import METRIC, ROOT, TRAIN, assess_file, sessions
from evaluation import (
    ACTIONS,
    SEED,
    SPLIT_VERSION,
    Event,
    Window,
    split_session,
    validation_windows,
    weighted_recall,
)
from predict import predict
from ranker import PARAMS, ROUNDS, TRAIN_EVERY, Ranker, fit_ranker
from submission import validate_submission, write_submission

from kumite.experiments import ExperimentRecord
from kumite.paths import REPO_ROOT
from kumite.provenance import code_version, file_sha256
from kumite.quality_gates import (
    Artifact,
    SelectionPlan,
    SourceSpec,
    publish_export,
    publish_gated_cv,
    reserve_cv,
    select_runs,
    write_selection,
)
from kumite.runs import RunStore, validate_run_id

CHANGES = {
    "B00": "直近順と行動別人気順を基準に固定",
    "E01": "40候補を行動別 LambdaRank で再ランキング",
    "E02": "教師グループを固定したまま候補を40件から100件へ拡大",
    "E03": "100候補のまま41〜100位に初めて正解が入る教師を追加",
    "E04": "候補商品の最終イベントから予測時点までの経過秒を追加",
}


def popularity(window: Window) -> dict[str, dict[int, int]]:
    counts = (
        pl.scan_parquet(TRAIN / "*.parquet")
        .filter((pl.col("start") >= window.start) & (pl.col("end") < window.end))
        .select("events")
        .explode("events")
        .unnest("events")
        .group_by("type", "aid")
        .len()
        .collect(engine="streaming")
    )
    return {
        action: dict(
            counts.filter(pl.col("type") == action)
            .sort(["len", "aid"], descending=[True, False])
            .head(100)
            .select("aid", "len")
            .iter_rows()
        )
        for action in ACTIONS
    }


def train(method: Method, query_window: Window) -> dict[str, Ranker]:
    if method.training_hit_width == 0:
        return {}
    teacher = query_window.past_week()
    popular = popularity(teacher.past_week())
    return {
        action: fit_ranker(sessions(teacher, every=TRAIN_EVERY), action, popular[action], method)
        for action in ACTIONS
    }


@dataclass(frozen=True)
class Context:
    plan: SelectionPlan
    config: Path
    windows: tuple[Window, Window]
    data_hash: str
    split_hash: str

    @classmethod
    def load(cls, config: Path) -> "Context":
        with config.open("rb") as handle:
            raw = tomllib.load(handle)
        plan = SelectionPlan(tuple(raw["candidates"]), float(raw["tolerance"]))
        latest = int(
            pl.scan_parquet(TRAIN / "*.parquet").select(pl.col("end").max()).collect().item()
        )
        windows = validation_windows(latest)
        data_hash = file_sha256(ROOT / "data" / "train.jsonl")
        manifest = json.loads((TRAIN / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != data_hash:
            raise ValueError("Parquet の変換元と公式学習ファイルが一致しません")
        split_hash = hashlib.sha256(
            json.dumps(
                [data_hash, SPLIT_VERSION, SEED, [(w.start, w.end) for w in windows]]
            ).encode()
        ).hexdigest()
        return cls(plan, config, windows, data_hash, split_hash)

    def assess(self, artifact: Artifact, record: ExperimentRecord) -> tuple[list[float], float]:
        expected = self.data_hash, self.split_hash, METRIC
        if (record.data_hash, record.split_hash, record.metric) != expected:
            raise ValueError("OOF の評価条件が正本と一致しません")
        return assess_file(artifact("oof.csv.gz"), self.windows)

    def source(self, method_name: str) -> SourceSpec:
        local = ROOT.relative_to(REPO_ROOT).as_posix()
        return SourceSpec(
            REPO_ROOT,
            (
                *(
                    f"{local}/src/{name}.py"
                    for name in (
                        "workflow",
                        "dataset",
                        "evaluation",
                        "candidates",
                        "cached_inference",
                        "prepare",
                        "predict",
                        "ranker",
                        "submission",
                    )
                ),
                *(
                    f"src/kumite/{name}.py"
                    for name in (
                        "__init__",
                        "crossfit",
                        "experiments",
                        "paths",
                        "provenance",
                        "quality_gates",
                        "runs",
                        "source_audit",
                    )
                ),
                f"{local}/quality-gates.toml",
                self.config.relative_to(REPO_ROOT).as_posix(),
                "pyproject.toml",
                "uv.lock",
            ),
            {
                "method": method_name,
                "candidates": asdict(METHODS[method_name]),
                "ranker": PARAMS,
                "rounds": ROUNDS,
                "train_every": TRAIN_EVERY,
                "seed": SEED,
                "split": SPLIT_VERSION,
            },
            (f"{local}/src", "src"),
        )


def run_cv(context: Context, store: RunStore, run_id: str, method_name: str) -> None:
    started = perf_counter()
    method = METHODS[method_name]
    source = context.source(method_name)
    with reserve_cv(store, run_id, source, {}, context.plan) as (reservation, start):
        output = ROOT / "outputs" / f".{run_id}-oof.csv.gz"
        try:
            stats: list[dict[str, object]] = []
            fold_scores: list[float] = []
            total = {action: (0, 0) for action in ACTIONS}
            with gzip.open(output, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
                writer = csv.writer(handle)
                for fold, window in enumerate(context.windows):
                    rankers = train(method, window)
                    print(f"{run_id} fold {fold + 1}/{len(context.windows)} 推論開始", flush=True)
                    popularity_by_action = popularity(window.past_week())
                    for_prediction, for_scoring = tee(
                        (session, split_session(session, events))
                        for session, events in sessions(window)
                    )
                    prefixes = ((session, prefix) for session, (prefix, _) in for_prediction)
                    truths = (
                        (session, action, truth[action])
                        for session, (_, truth) in for_scoring
                        for action in ACTIONS
                    )
                    predictions = predict(prefixes, method, popularity_by_action, rankers)
                    diagnostics = {
                        action: {"hits": 0, "candidate_hits_at_20": 0, "denominator": 0}
                        for action in ACTIONS
                    }
                    for expected, prediction in zip(truths, predictions, strict=True):
                        session, action, top, pool = prediction
                        truth_session, truth_action, targets = expected
                        if (session, action) != (truth_session, truth_action):
                            raise ValueError("予測と評価キーが一致しません")
                        action_counts = diagnostics[action]
                        action_counts["denominator"] += min(20, len(targets))
                        action_counts["hits"] += len(targets.intersection(top))
                        action_counts["candidate_hits_at_20"] += min(
                            20, len(targets.intersection(pool))
                        )
                        writer.writerow((fold, session, action, " ".join(map(str, top))))
                    stats.append(
                        {
                            "evaluation": diagnostics,
                            "training": {
                                action: {
                                    "groups": model.groups,
                                    "rows": model.rows,
                                    "positives": model.positives,
                                }
                                for action, model in rankers.items()
                            },
                        }
                    )
                    counts = {
                        action: (row["hits"], row["denominator"])
                        for action, row in diagnostics.items()
                    }
                    fold_scores.append(weighted_recall(counts))
                    total = {
                        action: (hits + counts[action][0], denominator + counts[action][1])
                        for action, (hits, denominator) in total.items()
                    }
                    print(f"{run_id} fold {fold + 1}/{len(context.windows)} 推論完了", flush=True)
            score = weighted_recall(total)
            record = ExperimentRecord(
                competition_id="040",
                run_id=run_id,
                metric=METRIC,
                greater_is_better=True,
                data_version="official-v1",
                data_hash=context.data_hash,
                split_version=SPLIT_VERSION,
                split_hash=context.split_hash,
                seed=SEED,
                hypothesis=f"{method_name} の事前比較計画",
                changed_one_thing=CHANGES[method_name],
                model="lambdarank" if method.training_hit_width else "recency-popularity",
                params={"method": method_name},
                fold_scores=fold_scores,
                oof_score=score,
                runtime_sec=perf_counter() - started,
                code_version=code_version(REPO_ROOT),
            )

            def write(directory: Path) -> None:
                output.replace(directory / "oof.csv.gz")
                (directory / "diagnostics.json").write_text(
                    json.dumps(stats) + "\n", encoding="utf-8"
                )

            publish_gated_cv(
                store,
                reservation,
                record,
                write,
                source=source,
                before_training=start,
                assess=context.assess,
            )
        finally:
            output.unlink(missing_ok=True)
    print(f"{run_id} {METRIC} {score:.8f}")


def official_test_prefixes() -> Iterator[tuple[int, list[Event]]]:
    parent = ROOT / "data" / "parquet" / "test"
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != file_sha256(ROOT / "data" / "test.jsonl"):
        raise ValueError("テスト Parquet の変換元が公式テストファイルと一致しません")
    for part in sorted(parent.glob("*.parquet")):
        for session, events in pl.read_parquet(part, columns=["session", "events"]).iter_rows():
            yield int(session), cast(list[Event], events)


def export(context: Context, store: RunStore, run_id: str, receipt: Path) -> None:
    # 選定記録の内容と現在のソースは publish_export が照合する。
    candidate = json.loads(receipt.read_text(encoding="utf-8"))["winner"]
    method_name = str(store.record(candidate).params["method"])
    method = METHODS[method_name]
    template = ROOT / "data" / "sample_submission.csv"
    after_training = context.windows[-1].end
    window = Window(after_training, after_training + 1)

    def write(directory: Path, winner: ExperimentRecord) -> None:
        if winner.params["method"] != method_name:
            raise ValueError("選定済みの手法が一致しません")
        rankers = train(method, window)
        for action, ranker in rankers.items():
            ranker.model.save_model(directory / f"{action}.txt")
        predictions = predict(
            official_test_prefixes(), method, popularity(window.past_week()), rankers
        )
        write_submission(
            directory / "submission.csv",
            template,
            ((f"{session}_{action}", top) for session, action, top, _ in predictions),
        )

    publish_export(
        store,
        run_id,
        plan=context.plan,
        receipt=receipt,
        assess=context.assess,
        source=context.source(method_name),
        write=write,
        required_outputs=("submission.csv",),
        validate_outputs=lambda path: validate_submission(path / "submission.csv", template),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "selection-p4.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv")
    cv.add_argument("run_id")
    cv.add_argument("method", choices=METHODS)
    select = commands.add_parser("select")
    select.add_argument("receipt", type=Path)
    output = commands.add_parser("export")
    output.add_argument("run_id")
    output.add_argument("receipt", type=Path)
    args = parser.parse_args()
    context = Context.load(args.plan.resolve())
    store = RunStore(ROOT)
    if args.command == "select":
        write_selection(args.receipt, select_runs(store, context.plan, context.assess))
    else:
        validate_run_id(args.run_id)
        if args.command == "cv":
            run_cv(context, store, args.run_id, args.method)
        else:
            export(context, store, args.run_id, args.receipt)


if __name__ == "__main__":
    main()
