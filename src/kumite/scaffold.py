"""テンプレートからコンペ用ディレクトリを生成する。"""

from dataclasses import asdict
from pathlib import Path
from string import Template

from kumite.catalog import Competition

# 中身は空だが、最初から存在してほしいディレクトリ。
TRACKED_DIRS = ("src", "notebooks")
UNTRACKED_DIRS = ("data", "outputs")


def create_competition_dir(
    competition: Competition, competitions_dir: Path, template_dir: Path
) -> Path:
    target = competitions_dir / competition.dir_name
    if target.exists():
        raise FileExistsError(f"既に存在します: {target}")

    variables = {key: str(value) for key, value in asdict(competition).items()}
    variables["number"] = f"{competition.number:03d}"

    target.mkdir(parents=True)
    for template in sorted(template_dir.iterdir()):
        text = Template(template.read_text(encoding="utf-8")).substitute(variables)
        (target / template.name).write_text(text, encoding="utf-8", newline="\n")
    for name in TRACKED_DIRS:
        (target / name).mkdir()
        (target / name / ".gitkeep").touch()
    for name in UNTRACKED_DIRS:
        (target / name).mkdir()
    return target
