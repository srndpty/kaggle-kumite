"""提出時のファイル配置だけが変わった場合に保存 OOF を再検証する。"""

import json
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

from data import ROOT
from train import REPO, assess, plan, source

from kumite.provenance import code_version
from kumite.quality_gates import publish_gated_cv, reserve_cv, verify_cv_run
from kumite.runs import CV_EVIDENCE, RunStore


def refresh() -> None:
    store = RunStore(ROOT)
    allowed_changes = {
        (ROOT / name).relative_to(REPO).as_posix()
        for name in ("src/export.py", "selection-plan.json")
    }
    for run_id in plan().candidates:
        original_id = run_id.removesuffix("-pkg")
        if original_id == run_id:
            msg = "移行先 run ID に -pkg が必要です"
            raise ValueError(msg)
        original = verify_cv_run(store, original_id, assess)
        previous = json.loads(
            store.artifact_path(original_id, CV_EVIDENCE).read_text(encoding="utf-8")
        )
        previous_source = json.loads(previous["source"])
        current = source(str(original.params["trial"]))
        current_source = json.loads(current.snapshot())
        for payload in (previous_source, current_source):
            for path in allowed_changes:
                del payload["files"][path]
        if previous_source != current_source:
            msg = "提出時のファイル配置以外にソースまたは設定の変更があります"
            raise ValueError(msg)
        names = tuple(sorted(original.artifacts.keys() - {CV_EVIDENCE}))
        inputs = store.input_hashes(original_id, *names)
        with reserve_cv(store, run_id, current, inputs, plan()) as (reservation, started):
            record = replace(
                original,
                run_id=run_id,
                created_at="",
                artifacts={},
                inputs=inputs,
                runtime_sec=None,
                code_version=code_version(REPO),
                hypothesis="提出時の変更後に既存 OOF を照合",
                changed_one_thing="Kaggle Dataset の展開済みファイルに対応",
            )

            def write(
                path: Path, names: tuple[str, ...] = names, original_id: str = original_id
            ) -> None:
                for name in names:
                    copyfile(store.artifact_path(original_id, name), path / name)

            print(
                publish_gated_cv(
                    store,
                    reservation,
                    record,
                    write,
                    source=current,
                    before_training=started,
                    assess=assess,
                ),
            )


if __name__ == "__main__":
    refresh()
