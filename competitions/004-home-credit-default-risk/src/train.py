"""事前登録した実験を新規runとして実行する。"""

import argparse

from pipeline import EXPERIMENTS, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="第0foldのみ。選定対象外")
    args = parser.parse_args()
    run(args.run_id, args.experiment, args.seed, args.benchmark)
