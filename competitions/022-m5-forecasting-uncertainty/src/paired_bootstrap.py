"""丸めの効果を、同一商品の全店舗・全CV窓をまとめたペア bootstrap で確認する。"""

import argparse
import json
from dataclasses import asdict

import numpy as np
from metric import evaluation_inputs, series_losses
from train import DATA, ORIGINS, STORE, hierarchy_for, input_hash

from kumite.bootstrap import group_bootstrap_diff
from kumite.m5 import LEVELS, VALIDATION_FILE, load_sales

N_BOOTSTRAP = 10_000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("other")
    args = parser.parse_args()
    base, other = STORE.record(args.base), STORE.record(args.other)
    if (
        (base.data_version, other.data_version) != ("E01", "E05")
        or base.data_hash != other.data_hash
        or base.data_hash != input_hash(VALIDATION_FILE)
        or (base.split_version, base.split_hash) != (other.split_version, other.split_hash)
    ):
        raise ValueError("同じデータ・分割での E01、E05 の CV run を指定してください")

    data = load_sales(DATA)
    hierarchy = hierarchy_for(data)
    bottom = hierarchy.levels == len(LEVELS) - 1
    # 共通階層の最下層は item_id、store_id の順。同じ商品を店舗間でもまとめる。
    keys = data.series.sort_values(["item_id", "store_id"])
    groups = keys["item_id"].to_numpy()
    contributions = np.zeros((2, len(groups)))
    for origin in ORIGINS:
        actual, weight, scale = evaluation_inputs(data, hierarchy, origin)
        base_pred, other_pred = (
            STORE.load_array(run.run_id, f"d_{origin}.npy") for run in (base, other)
        )
        if not np.array_equal(base_pred[~bottom], other_pred[~bottom]):
            raise ValueError("最下層以外の予測が一致しません")
        for index, pred in enumerate((base_pred, other_pred)):
            contributions[index] += series_losses(
                actual[bottom], pred[bottom], weight[bottom], scale[bottom]
            ).mean(axis=1)

    # 共通関数は行平均を返すため、行数を掛け、窓と階層の数で割る。
    contributions *= len(groups) / (len(ORIGINS) * len(LEVELS))
    diff = group_bootstrap_diff(
        contributions[0],
        contributions[1],
        groups,
        transform=lambda x: x,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    np.testing.assert_allclose(diff.point, other.oof_score - base.oof_score, atol=1e-12, rtol=0)
    print(
        json.dumps(
            {
                "base": base.run_id,
                "other": other.run_id,
                "items": len(np.unique(groups)),
                "n_bootstrap": N_BOOTSTRAP,
                "seed": SEED,
                **asdict(diff),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
