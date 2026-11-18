"""実行: uv run python competitions/002-porto-seguro-safe-driver-prediction/src/train.py E01"""

import argparse

from pipeline import EXPERIMENTS, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Porto Seguro の CV と提出ファイル作成")
    parser.add_argument("run_id")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.run_id, args.experiment or args.run_id, args.seed)
