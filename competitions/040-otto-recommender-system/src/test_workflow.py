import gzip
from pathlib import Path

import dataset
import numpy as np
import pytest
import workflow
from cached_inference import CachedRanker
from candidates import METHODS, rank_features
from evaluation import Event, validation_windows
from ranker import fit_ranker
from submission import validate_submission, write_submission
from workflow import ROOT, Context

from kumite.quality_gates import SelectionPlan, select_runs, verify_export, write_selection
from kumite.runs import RunStore


def test_saved_oof_is_rescored_and_extra_query_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_windows(999)[-1]
    repeated: list[dataset.Session] = [
        (1, [{"aid": 1, "ts": 1, "type": "orders"}, {"aid": 2, "ts": 2, "type": "orders"}]),
        (2, [{"aid": 1, "ts": 1, "type": "carts"}, {"aid": 2, "ts": 2, "type": "carts"}]),
        (3, [{"aid": 1, "ts": 1, "type": "clicks"}, {"aid": 2, "ts": 2, "type": "clicks"}]),
    ]
    monkeypatch.setattr(dataset, "sessions", lambda _: iter(repeated))
    predictions = tmp_path / "oof.csv.gz"
    with gzip.open(predictions, "wt", encoding="utf-8", newline="") as output:
        for session, _ in repeated:
            for action in ("clicks", "carts", "orders"):
                output.write(f"0,{session},{action},2\n")
    folds, score = dataset.assess_file(predictions, (window,))
    assert folds == [pytest.approx(1)]
    assert score == pytest.approx(1)
    with gzip.open(predictions, "at", encoding="utf-8", newline="") as output:
        output.write("0,4,clicks,2\n")
    with pytest.raises(ValueError, match="余分"):
        dataset.assess_file(predictions, (window,))


@pytest.mark.parametrize("method_name", ["E01", "E04"])
def test_ranker_trains_and_returns_unique_ordered_subset(method_name: str) -> None:
    events: list[Event] = [
        {"aid": 3, "ts": 1, "type": "clicks"},
        {"aid": 3, "ts": 2, "type": "orders"},
    ]
    popularity = dict.fromkeys(range(100), 2)
    method = METHODS[method_name]
    model = fit_ranker(((i * 100, events) for i in range(40)), "orders", popularity, method)
    assert (model.groups, model.rows, model.positives) == (40, 40 * method.width, 40)
    query, top, pool = CachedRanker(model, method, popularity).rank([(999, events[:1])])[0]
    features = np.asarray(
        rank_features(events[:1], pool, popularity, elapsed_feature=method.elapsed_feature),
        dtype=np.float32,
    )
    scores = np.asarray(model.model.predict(features, num_threads=1))
    assert top == [pool[i] for i in np.argsort(-scores, kind="stable")[:20]]
    assert query == 999
    assert len(top) == len(set(top)) == 20
    assert set(top) <= set(pool)


def test_submission_rejects_duplicate_aids_and_missing_template_rows(tmp_path: Path) -> None:
    template = tmp_path / "sample.csv"
    output = tmp_path / "submission.csv"
    template.write_text("session_type,labels\n7_clicks,1\n7_carts,1\n", encoding="utf-8")
    write_submission(output, template, [("7_clicks", [2]), ("7_carts", [3])])
    validate_submission(output, template)
    write_submission(output, template, [("7_clicks", [2, 2]), ("7_carts", [3])])
    with pytest.raises(ValueError, match="商品リスト"):
        validate_submission(output, template)
    with pytest.raises(ValueError, match="shorter"):
        write_submission(output, template, [("7_clicks", [2])])


def test_source_spec_passes_dependency_audit() -> None:
    context = Context(
        SelectionPlan(("E01",)),
        ROOT / "selection-p4.toml",
        validation_windows(999),
        "test-data",
        "test-split",
    )
    assert '"method": "E01"' in context.source("E01").snapshot()


def test_gated_cv_selection_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "quality-gates.toml").write_text("", encoding="utf-8")
    (data / "sample_submission.csv").write_text(
        "session_type,labels\n999_clicks,1\n999_carts,1\n999_orders,1\n", encoding="utf-8"
    )
    repeated: list[dataset.Session] = [
        (sid, [{"aid": 1, "ts": 1, "type": action}, {"aid": 2, "ts": 2, "type": action}])
        for sid, action in enumerate(("clicks", "carts", "orders"))
    ]
    population = {action: {2: 1} for action in ("clicks", "carts", "orders")}
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "popularity", lambda _: population)
    monkeypatch.setattr(workflow, "sessions", lambda _: iter(repeated))
    monkeypatch.setattr(dataset, "sessions", lambda _: iter(repeated))
    monkeypatch.setattr(
        workflow, "official_test_prefixes", lambda: iter([(999, repeated[0][1][:1])])
    )
    context = Context(
        SelectionPlan(("B00",)),
        ROOT / "selection-p4.toml",
        validation_windows(999),
        "test-data",
        "test-split",
    )
    monkeypatch.setattr(Context, "source", lambda _, method: _test_source(method))
    store = RunStore(tmp_path)
    workflow.run_cv(context, store, "B00", "B00")
    assert store.record("B00").oof_score == pytest.approx(1)
    receipt = tmp_path / "selection.json"
    write_selection(receipt, select_runs(store, context.plan, context.assess))
    workflow.export(context, store, "F00", receipt)
    exported = verify_export(RunStore(tmp_path / "outputs" / "exports"), "F00")
    assert exported.metric == "cv_reference:WeightedRecall@20"


def _test_source(method: str) -> workflow.SourceSpec:
    return workflow.SourceSpec(ROOT, ("selection-p4.toml",), {"method": method})
