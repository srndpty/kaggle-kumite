"""宣言した Python ソースの静的 import を、指定されたリポジトリ内で照合する。"""

import ast
import tokenize
from collections.abc import Iterator
from pathlib import Path


def _module_files(base: Path, parts: tuple[str, ...]) -> Iterator[Path]:
    for end in range(1, len(parts) + 1):
        package = base.joinpath(*parts[:end], "__init__.py")
        if package.is_file():
            yield package
    if parts:
        module = base.joinpath(*parts).with_suffix(".py")
        if module.is_file():
            yield module


def _imports(
    tree: ast.AST, source: Path, import_roots: tuple[Path, ...]
) -> Iterator[tuple[Path, tuple[str, ...]]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for base in import_roots:
                    yield base, tuple(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            parts = tuple(node.module.split(".")) if node.module else ()
            bases: tuple[Path, ...]
            if node.level > 0:
                if node.level > len(source.parents):
                    continue
                bases = (source.parents[node.level - 1],)
            else:
                bases = import_roots
            for base in bases:
                yield base, parts
                for alias in node.names:
                    if alias.name != "*":
                        yield base, (*parts, alias.name)


def verify_local_imports(root: Path, declared: set[str], import_roots: tuple[str, ...]) -> None:
    """静的 import で参照するローカルモジュールとパッケージ初期化ファイルを検査する。"""
    if not import_roots:
        raise ValueError("Python の検索ルートがありません")
    roots = tuple((root / name).resolve() for name in import_roots)
    for base in roots:
        if not base.is_relative_to(root) or not base.is_dir():
            raise ValueError(f"Python の検索ルートが不正です: {base}")

    for name in sorted(declared):
        source = root / name
        if source.suffix != ".py":
            continue
        with tokenize.open(source) as file:
            tree = ast.parse(file.read(), filename=name)
        for base, parts in _imports(tree, source, roots):
            if not base.is_relative_to(root):
                raise ValueError(f"{name}: 相対 import がソースルートの外を参照しています")
            for imported in _module_files(base, parts):
                relative = imported.relative_to(root).as_posix()
                if relative not in declared:
                    raise ValueError(f"{name}: 未宣言のローカル import: {relative}")
