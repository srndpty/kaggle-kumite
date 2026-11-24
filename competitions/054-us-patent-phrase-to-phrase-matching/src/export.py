"""OOF で選定した設定を全行で再学習する提出 Notebook を生成する。"""

import argparse
from pathlib import Path

from train import TRIALS

from kumite.kaggle_notebook import write_cpu_notebook
from kumite.runs import RunStore

ROOT = Path(__file__).resolve().parents[1]
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


def inference_source(features: tuple[str, ...]) -> str:
    checks = Path(__file__).resolve().parents[3] / "src" / "kumite" / "checks.py"
    inputs = (ROOT / "src" / "inputs.py").read_text(encoding="utf-8")
    model = (ROOT / "src" / "model.py").read_text(encoding="utf-8")
    return "\n".join(
        (
            checks.read_text(encoding="utf-8"),
            inputs.replace("from kumite.checks import validate_regression", ""),
            model,
            PREDICT,
            f"FEATURES = {features!r}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner")
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    store = RunStore(ROOT)
    if args.baseline:
        record = store.record(f"E01{args.run_suffix}")
        name = "baseline"
    else:
        records = [store.record(f"{trial}{args.run_suffix}") for trial in TRIALS]
        if len({(record.data_hash, record.split_hash) for record in records}) != 1:
            raise ValueError("同じデータと分割の run が必要です")
        record = max(records, key=lambda item: item.oof_score)
        name = "final"
    source = inference_source(tuple(record.params["features"]))
    write_cpu_notebook(
        args.output,
        source + BODY,
        kernel_id=f"{args.owner}/kumite-patent-{name}",
        title=f"Kumite Patent {name.title()}",
        competition=SLUG,
    )
    print(record.run_id)


if __name__ == "__main__":
    main()
