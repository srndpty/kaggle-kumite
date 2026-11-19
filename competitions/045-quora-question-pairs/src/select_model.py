"""事前登録した OOF の規則で最終提出 run を決める。"""

import json

from predict import ROOT

from kumite.experiments import ExperimentRecord, find_record, read_records


def comparison_key(record: ExperimentRecord) -> tuple[str, str, str, int]:
    return record.data_hash, record.split_version, record.split_hash, record.seed


def main() -> None:
    records = read_records(ROOT / "experiments.jsonl")
    candidates = [find_record(records, run_id) for run_id in ("E01", "E02-cv", "E03", "E04", "E05")]
    if len({comparison_key(item) for item in candidates}) != 1 or candidates[0].seed != 42:
        raise ValueError("seed 42 の候補が同じデータ・分割を使用していません")
    first, second, *_ = sorted(candidates, key=lambda item: (item.oof_score, item.run_id))
    scores = {item.run_id: item.oof_score for item in candidates}
    winner = first
    means = {}
    if second.oof_score - first.oof_score < 0.001:
        repeats = [
            find_record(records, f"{item.run_id.removesuffix('-cv')}-seed43")
            for item in (first, second)
        ]
        if (
            comparison_key(repeats[0]) != comparison_key(repeats[1])
            or repeats[0].seed != 43
            or repeats[0].data_hash != first.data_hash
        ):
            raise ValueError("seed 43 の再測定条件が一致しません")
        means = {
            item.run_id: (item.oof_score + repeat.oof_score) / 2
            for item, repeat in zip((first, second), repeats, strict=True)
        }
        winner = min((first, second), key=lambda item: (means[item.run_id], item.run_id))
    print(
        json.dumps(
            {
                "selected_run": winner.run_id.removesuffix("-cv"),
                "selection_evidence_run": winner.run_id,
                "seed42": scores,
                "two_seed_mean": means,
            }
        )
    )


if __name__ == "__main__":
    main()
