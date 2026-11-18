"""小規模CPU学習からrun成果物の検査まで。外部データは不要。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pipeline
import pytest
from dataset import KEY
from select_model import validate_record

from kumite.runs import RunStore


def test_training_publication_and_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    train = pd.DataFrame(
        {
            KEY: np.arange(120),
            "TARGET": (x > 0).astype(int),
            "x": x,
            "cat": np.where(x > 0.2, "a", "b"),
        }
    )
    test = train.drop(columns="TARGET").iloc[:12].assign(SK_ID_CURR=np.arange(500, 512))
    sample = test[[KEY]].assign(TARGET=0.0)
    input_file = tmp_path / "input.csv"
    train.to_csv(input_file, index=False)
    store = RunStore(tmp_path)
    monkeypatch.setattr(pipeline, "STORE", store)
    monkeypatch.setattr(pipeline, "DATA_FILES", (input_file,))
    monkeypatch.setattr(pipeline, "load_data", lambda: (train, test, sample))
    original = pipeline.parameters
    monkeypatch.setattr(
        pipeline,
        "parameters",
        lambda seed: {
            **original(seed),
            "num_boost_round": 5,
            "early_stopping": None,
            "min_data_in_leaf": 5,
        },
    )
    pipeline.run("smoke", "E01")
    pred = store.load_array("smoke", "oof.npy")
    assert pred.shape == (120,)
    assert np.isfinite(pred).all()
    assert store.load_array("smoke", "test.npy").shape == (12,)
    record = store.record("smoke")
    assert len(record.fold_scores) == 3
    assert record.params["definition"]["history"] == ""
    assert json.loads(store.artifact_path("smoke", "features.json").read_text(encoding="utf-8"))[
        "features"
    ] == ["x", "cat"]
    infos = json.loads(store.artifact_path("smoke", "fold_info.json").read_text(encoding="utf-8"))
    for info in infos:
        preprocessing = info["preprocessing"]
        assert preprocessing["categorical_columns"] == ["cat"]
        assert preprocessing["category_counts"] == {"cat": 2}
        assert preprocessing["dtypes"] == {"x": "float32", "cat": "category"}
        assert preprocessing["unknown_category"] == "missing"
    assert store.orphans() == []
    with pytest.raises(FileExistsError):
        pipeline.run("smoke", "E01")
    with pytest.raises(ValueError, match="実験定義"):
        validate_record(record, "E02", 42)
    path = store.artifact_path("smoke", "submission.csv")
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="ハッシュ"):
        store.artifact_path("smoke", "submission.csv")
