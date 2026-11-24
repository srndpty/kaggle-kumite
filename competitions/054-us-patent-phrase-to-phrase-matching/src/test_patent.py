"""固定分割、類似度、Notebook 推論の回帰テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from evidence import ROOT, inference_sources
from export import inference_source
from inputs import read_rows, submission
from model import PatentRidge, pearson
from scipy.stats import pearsonr
from split import make_split


def test_split_keeps_anchor_and_reversed_pair_together_across_cpc() -> None:
    pairs = [("a", "b", "A01"), ("a", "c", "B01"), ("b", "a", "C01")]
    pairs += [(str(i), "z", "A01") for i in range(8)]
    rows = pd.DataFrame(pairs, columns=["anchor", "target", "context"]).assign(id=range(len(pairs)))
    split = make_split(rows)
    assert split["id"].is_unique
    assert split.loc[:2, "fold"].nunique() == 1
    assert split.loc[:2, "component"].nunique() == 1
    assert split["fold"].nunique() == 5
    assert split.groupby("component")["fold"].nunique().eq(1).all()


def test_split_does_not_link_a_chain_of_distinct_pairs() -> None:
    pairs = [("a", "b"), ("b", "c"), ("c", "d")] + [(str(i), "z") for i in range(8)]
    rows = pd.DataFrame(pairs, columns=["anchor", "target"]).assign(id=range(len(pairs)))
    assert make_split(rows).loc[:2, "component"].nunique() == 3


def test_peer_similarity_uses_only_other_rows_of_same_anchor_and_cpc() -> None:
    rows = pd.DataFrame(
        {
            "anchor": ["motor"] * 4 + ["clamp"],
            "target": ["motor", "motors", "motor ring", "ring", "clamp"],
            "context": ["A01", "A01", "A01", "B01", "A01"],
        }
    )
    model = PatentRidge(("char", "cpc", "interaction", "peer_similarity"))
    matrix = model.matrix(rows, fit=True).toarray()
    cosine = matrix[:, 5]
    assert np.ptp(cosine[:3]) > 0
    np.testing.assert_allclose(
        matrix[:, -1],
        [
            (cosine[1] + cosine[2]) / 2,
            (cosine[0] + cosine[2]) / 2,
            (cosine[0] + cosine[1]) / 2,
            0,
            0,
        ],
    )
    assert model.matrix(rows.iloc[:1]).toarray()[0, -1] == 0


def test_pearson() -> None:
    y = np.array([0.0, 0.5, 1.0])
    pred = np.array([0.25, 0.2, 0.9])
    assert pearson(y, pred) == pytest.approx(pearsonr(y, pred)[0])
    with pytest.raises(ValueError, match="Pearson"):
        pearson(y, np.zeros(3))


def test_notebook_matches_local_inference_with_unknown_cpc(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text(
        "id,anchor,target,context,score\n1,motor,engine,B01,0.5\n"
        "2,motor,motors,B02,1\n3,clamp,ring,A01,0\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "id,anchor,target,context\n5,motor,motoring,Y01\n4,clamp,ring,C01\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text("id,score\n5,0\n4,0\n", encoding="utf-8")
    features = ("char", "cpc", "interaction", "peer_similarity")
    model = PatentRidge(features)
    model.fit(read_rows(train_path, labeled=True))
    test = read_rows(test_path)
    expected = submission(test, model.predict(test))
    namespace: dict[str, object] = {}
    exec(inference_source(features, inference_sources(ROOT)), namespace)
    output = tmp_path / "submission.csv"
    predict = namespace["predict_directory"]
    assert callable(predict)
    predict(tmp_path, output, features)
    actual = pd.read_csv(output, dtype={"id": str})
    pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(ValueError):
        submission(test, np.array([0.5, np.inf]))
    with pytest.raises(ValueError):
        submission(test, np.array([0.5, 1.1]))
