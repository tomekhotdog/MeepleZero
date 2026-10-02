"""Import layering is enforced here, deterministically, from the source AST.

Why: the package is layered ``core -> game -> agents/nn -> training/web -> cli``
so each layer can be reasoned about (and tested) without the ones above it.
An upward import (e.g. ``game`` reaching into ``agents``) is the first step
towards cycles and cross-module reasoning, so it must fail CI, not review.

Within a tier, sibling edges are allowed only one way, matching the code as
it stands: ``agents -> nn`` (MCTS evaluates the net) but never ``nn -> agents``;
``web -> training`` (the dashboard reads ``TrainingRun`` dirs) but never
``training -> web``.

Every import counts -- module level, inside functions, and inside
``if TYPE_CHECKING:`` blocks. The rule is about dependency direction, and a
type-only upward import is still an upward dependency (and the usual first
step before a runtime one).
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = "carcassonne"
PKG_DIR = Path(__file__).resolve().parent.parent / PKG

_ALL = frozenset({"core", "game", "agents", "nn", "training", "web", "cli"})

# layer -> layers (subpackages of carcassonne) it may import from.
ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset({"core"}),
    "game": frozenset({"core", "game"}),
    "nn": frozenset({"core", "game", "nn"}),
    "agents": frozenset({"core", "game", "nn", "agents"}),
    "training": frozenset({"core", "game", "nn", "agents", "training"}),
    "web": frozenset({"core", "game", "nn", "agents", "training", "web"}),
    "cli": _ALL,
}


def _module_parts(path: Path, pkg_dir: Path) -> list[str]:
    parts = [PKG, *path.relative_to(pkg_dir).with_suffix("").parts]
    return parts[:-1] if parts[-1] == "__init__" else parts


def _imported_modules(path: Path, pkg_dir: Path) -> list[tuple[int, str]]:
    """Every absolute module name imported anywhere in ``path`` (relative resolved)."""
    parts = _module_parts(path, pkg_dir)
    is_pkg = path.name == "__init__.py"
    package = parts if is_pkg else parts[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            found.append((node.lineno, name))
    return sorted(found)


def _layer_of(module: str) -> str | None:
    """``carcassonne.<layer>...`` -> layer; ``carcassonne`` itself -> "" ; else None."""
    parts = module.split(".")
    if parts[0] != PKG:
        return None
    return parts[1] if len(parts) > 1 else ""


def layering_violations(pkg_dir: Path = PKG_DIR) -> list[str]:
    violations: list[str] = []
    for path in sorted(pkg_dir.rglob("*.py")):
        src_parts = _module_parts(path, pkg_dir)
        src_layer = src_parts[1] if len(src_parts) > 1 else ""
        allowed = ALLOWED.get(src_layer, frozenset())
        for lineno, module in _imported_modules(path, pkg_dir):
            target = _layer_of(module)
            if target is None:
                continue
            if target not in allowed:
                rel = path.relative_to(pkg_dir.parent)
                violations.append(f"{rel}:{lineno} ({src_layer or PKG}) imports {module}")
    return violations


def test_every_package_module_is_in_a_known_layer() -> None:
    # A new top-level subpackage must be placed in the layer graph explicitly,
    # otherwise it would silently get the empty allow-set (or escape review).
    subpackages = {p.name for p in PKG_DIR.iterdir() if (p / "__init__.py").exists()}
    assert subpackages == set(ALLOWED), f"unlayered subpackages: {subpackages ^ set(ALLOWED)}"
    stray = [p.name for p in PKG_DIR.glob("*.py") if p.name != "__init__.py"]
    assert not stray, f"top-level modules outside any layer: {stray}"


def test_imports_respect_layering() -> None:
    violations = layering_violations()
    assert not violations, "layering violations (module:line imports target):\n  " + "\n  ".join(
        violations
    )


def test_resolves_relative_and_nested_imports(tmp_path: Path) -> None:
    # The checker itself must see relative, function-local and TYPE_CHECKING
    # imports, or the main test could pass vacuously.
    pkg = tmp_path / PKG
    for sub in ("core", "training"):
        (pkg / sub).mkdir(parents=True)
        (pkg / sub / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "core" / "a.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from carcassonne.training import x\n"
        "def f():\n"
        "    from ..training import y\n"
        "    import carcassonne.training.z\n"
        "from . import ok\n"
    )
    found = layering_violations(pkg)
    assert found == [
        "carcassonne/core/a.py:3 (core) imports carcassonne.training",
        "carcassonne/core/a.py:5 (core) imports carcassonne.training",
        "carcassonne/core/a.py:6 (core) imports carcassonne.training.z",
    ]
