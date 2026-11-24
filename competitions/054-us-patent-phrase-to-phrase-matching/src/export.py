"""OOF で選定した設定を全行で再学習する提出 Notebook を生成する。"""

import argparse
import json
from pathlib import Path

from evidence import ROOT, inference_sources
from select_model import select
from train import TRIALS

from kumite.kaggle_notebook import write_cpu_notebook

SLUG = "us-patent-phrase-to-phrase-matching"
BODY = """
test_file, = (
    path for path in Path('/kaggle/input').rglob('test.csv')
    if path.parent.name == 'us-patent-phrase-to-phrase-matching'
)
predict_directory(test_file.parent, Path('/kaggle/working/submission.csv'), FEATURES)
"""
PREDICT = """
def predict_directory(data, output, features):
    train, test = load_data(data)
    model = PatentRidge(features)
    model.fit(train)
    submission(test, model.predict(test)).to_csv(output, index=False)
"""


def inference_source(features: tuple[str, ...], sources: dict[str, bytes]) -> str:
    return "\n".join(
        (
            sources["checks.py"].decode("utf-8"),
            sources["inputs.py"]
            .decode("utf-8")
            .replace("from kumite.checks import validate_regression", ""),
            sources["model.py"].decode("utf-8"),
            PREDICT,
            f"FEATURES = {features!r}",
        )
    )


def verified_source(root: Path, path: Path, suffix: str, *, baseline: bool) -> tuple[str, str]:
    sources = inference_sources(root)
    frozen = None if baseline else json.loads(path.read_text(encoding="utf-8"))
    if frozen is not None:
        if not isinstance(frozen, dict) or not isinstance(frozen.get("run_suffix"), str):
            raise ValueError("選定記録の形式が不正です")
        suffix = frozen["run_suffix"]
    current = select(root, suffix, sources, baseline=baseline)
    if not baseline and frozen != current:
        raise ValueError("現在の選定結果が確定済みの選定記録と一致しません")
    run_id = current["selected"]
    features = TRIALS[run_id.removesuffix(suffix)][0]
    return run_id, inference_source(features, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--selection", type=Path, default=ROOT / "selection.json")
    args = parser.parse_args()
    if args.run_suffix and not args.baseline:
        parser.error("最終提出の run 接尾辞は選定記録から取得します")
    run_id, source = verified_source(ROOT, args.selection, args.run_suffix, baseline=args.baseline)
    name = "baseline" if args.baseline else "final"
    write_cpu_notebook(
        args.output,
        source + BODY,
        kernel_id=f"{args.owner}/kumite-patent-{name}",
        title=f"Kumite Patent {name.title()}",
        competition=SLUG,
    )
    print(run_id)


if __name__ == "__main__":
    main()
