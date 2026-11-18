"""事前登録した実験を実行する。"""

import argparse

from pipeline import EXPERIMENTS, train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--experiment", choices=EXPERIMENTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    train(a.run_id, a.experiment or a.run_id, a.seed, a.epochs)
