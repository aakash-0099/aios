#!/usr/bin/env python3
"""Import-graph and layering checker for the AIOS ``aios`` package.

Parses every ``.py`` file under ``aios/`` (tests and ``__pycache__`` excluded)
with :mod:`ast`, resolves every absolute and relative import -- including
imports nested inside functions, methods and ``try`` blocks -- to a
top-level ``aios`` package, then prints a package-to-package dependency table
and checks a fixed set of layering rules.

The layering rules live in the single data structure :data:`RULES` near the
top of this file; **that is the only place you need to edit to change the
policy.**  Each rule is a :class:`Rule` with one of three kinds:

``allow``
    the package may import only the listed packages (anything else is a
    violation); imports *within* the package itself are always allowed;
``forbid``
    the package may import everything *except* the listed packages;
``pair_forbid``
    a sub-package-to-sub-package restriction (``aios.agents.task`` may not
    import ``aios.agents.agent_manager``).

A rule subject is matched as a *package prefix*, so ``aios.agents`` also
governs ``aios.agents.task``, and a target matches a package and all of its
sub-packages (``aios.core`` covers ``aios.core.models``).

Every intra-``aios`` import is kept internally so sub-package rules can be
evaluated, but imports that stay inside one top-level package are excluded
from the printed table, the cycle check and the "not covered by a rule"
list -- they carry no layering information.  Cycles are therefore detected
over top-level package edges only.

Usage::

    python scripts/check_imports.py            # report only, always exit 0
    python scripts/check_imports.py --strict   # exit 1 if a rule is violated
    python scripts/check_imports.py --json     # dependency graph as JSON
    python scripts/check_imports.py --mermaid  # Mermaid 'graph LR' diagram

Stdlib only: ast, pathlib, json, argparse.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# >>> POLICY DEFINITION -- edit this block to change the layering rules <<<
# --------------------------------------------------------------------------

#: Top-level sub-packages of ``aios`` that take part in the layering rules.
KNOWN_PACKAGES: tuple[str, ...] = (
    "core",
    "kernel",
    "scheduler",
    "llm",
    "context",
    "memory",
    "storage",
    "tools",
    "security",
    "agents",
    "config",
    "communication",
    "monitoring",
    "sdk",
)


@dataclass(frozen=True)
class Rule:
    """One layering rule.

    ``kind``:
      * ``"allow"``     -- only ``targets`` may be imported (or listed in
                           ``exempt``; everything else is a violation).
      * ``"forbid"``    -- none of ``targets`` may be imported.
      * ``"pair_forbid"`` -- same as ``"forbid"`` but the subject is a
                           sub-package path (``aios.agents.task``) rather
                           than a top-level package.
    """

    kind: str
    subject: str
    targets: frozenset[str]
    reason: str
    exempt: frozenset[str] = field(default_factory=frozenset)


RULES: tuple[Rule, ...] = (
    Rule(
        kind="allow",
        subject="aios.core",
        targets=frozenset(),
        exempt=frozenset(),
        reason="core is the base layer and must not depend on any other aios package",
    ),
    Rule(
        kind="allow",
        subject="aios.config",
        targets=frozenset({"aios.core"}),
        exempt=frozenset(),
        reason="config may only build on core",
    ),
    Rule(
        kind="allow",
        subject="aios.storage",
        targets=frozenset({"aios.core"}),
        exempt=frozenset(),
        reason="storage may only build on core",
    ),
    Rule(
        kind="allow",
        subject="aios.memory",
        targets=frozenset({"aios.core", "aios.storage"}),
        exempt=frozenset(),
        reason="memory may only build on core and storage",
    ),
    Rule(
        kind="forbid",
        subject="aios.scheduler",
        targets=frozenset({"aios.kernel"}),
        reason="Scheduler is a standalone decision service; the Kernel orchestrates",
    ),
    Rule(
        kind="forbid",
        subject="aios.tools",
        targets=frozenset({"aios.kernel", "aios.scheduler"}),
        reason="tools is a leaf capability layer, wired by the Kernel",
    ),
    Rule(
        kind="forbid",
        subject="aios.llm",
        targets=frozenset({"aios.kernel", "aios.scheduler"}),
        reason="llm is a leaf provider layer, wired by the Kernel",
    ),
    Rule(
        kind="pair_forbid",
        subject="aios.agents.task",
        targets=frozenset({"aios.agents.agent_manager"}),
        reason="task and agent_manager stay decoupled; Kernel wires them together",
    ),
    Rule(
        kind="forbid",
        subject="aios.agents.agent_manager",
        targets=frozenset({"aios.kernel", "aios.scheduler"}),
        reason="agent_manager is wired by the Kernel, it must not import it",
    ),
)

# --------------------------------------------------------------------------
# END POLICY DEFINITION
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
AIOS_DIR = ROOT / "aios"
SKIP_DIR_NAMES = {"tests", "__pycache__"}


@dataclass(frozen=True)
class Edge:
    """A single resolved import, attributed to one source location.

    ``src``/``dst`` are the *top-level* packages (used for the table and the
    cycle check); ``src_module``/``dst_module`` keep the full dotted paths so
    sub-package rules such as ``aios.agents.task`` can be evaluated.
    """

    src: str  # top-level aios package, e.g. "aios.core"
    dst: str  # top-level aios package
    file: str  # path relative to repo root
    line: int
    src_module: str = ""  # full dotted module of the importing file
    dst_module: str = ""  # full dotted module of the imported name

    @property
    def sort_key(self) -> tuple[str, str, str, int]:
        return (self.src, self.dst, self.file, self.line)


@dataclass
class Violation:
    rule: str
    src: str
    dst: str
    file: str
    line: int
    reason: str


def iter_source_files(base: Path) -> list[Path]:
    """Yield every .py file under ``base`` outside tests/__pycache__ dirs."""
    out: list[Path] = []
    for path in sorted(base.rglob("*.py")):
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(base).parts[:-1]):
            continue
        if "__pycache__" in path.parts:
            continue
        out.append(path)
    return out


def top_level_package(module: str) -> str | None:
    """Return ``aios.<pkg>`` for a module path inside the aios package."""
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "aios":
        return "aios." + parts[1]
    return None


def package_of_file(path: Path) -> str:
    """Top-level aios package owning ``path`` (aios/__init__.py -> 'aios')."""
    rel = path.relative_to(AIOS_DIR).parts
    return "aios." + rel[0] if len(rel) > 1 else "aios"


def resolve_relative(
    module: str | None, level: int, pkg_parts: tuple[str, ...]
) -> str | None:
    """Resolve a relative import to an absolute dotted module path.

    ``level`` is the number of leading dots: 1 means the current package,
    2 its parent, and so on.  ``pkg_parts`` is the dotted package of the
    importing file (its module filename already stripped).
    """
    if level <= 0:
        return module
    if level > 1:
        base = list(pkg_parts[: len(pkg_parts) - (level - 1)])
    else:
        base = list(pkg_parts)
    if level - 1 > len(pkg_parts):
        base = []
    if module:
        base.extend(module.split("."))
    return ".".join(base) or None


def module_of_file(file: str) -> str:
    """Dotted module path of a repo-relative file under aios/, minus .py."""
    parts = list(Path(file).relative_to("aios").with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return "aios." + ".".join(parts)


def module_package_parts(path: Path) -> tuple[str, ...]:
    """Dotted package of the module in ``path`` (module filename stripped)."""
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)[:-1]  # drop filename or __init__.py
    return tuple(parts)


def extract_imports(path: Path) -> list[tuple[str, int]]:
    """Return ``(absolute_module, lineno)`` for every import in ``path``.

    Walks the whole tree so imports inside ``if``/``try``/functions/methods
    and ``__future__``-style conditional blocks are all captured.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    pkg_parts = module_package_parts(path)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                resolved = resolve_relative(node.module, node.level, pkg_parts)
                if resolved:
                    found.append((resolved, node.lineno))
            elif node.module:
                found.append((node.module, node.lineno))
    return found


def build_graph(base: Path) -> list[Edge]:
    """Parse every source file and resolve imports into edges.

    Edges are emitted for every intra-``aios`` import (including sub-packages
    such as ``aios.agents.task`` -> ``aios.agents.agent_manager``); the
    top-level table simply aggregates them.
    """
    edges: list[Edge] = []
    for path in iter_source_files(base):
        src_pkg = package_of_file(path)
        src_module = module_of_file(path.relative_to(ROOT).as_posix())
        rel = path.relative_to(ROOT).as_posix()
        for module, lineno in extract_imports(path):
            dst = top_level_package(module)
            if dst is None:
                continue
            edges.append(
                Edge(
                    src=src_pkg,
                    dst=dst,
                    file=rel,
                    line=lineno,
                    src_module=src_module,
                    dst_module=module,
                )
            )
    # de-duplicate identical (src, dst, file, line) records
    seen: set[tuple[str, str, str, int]] = set()
    unique: list[Edge] = []
    for e in sorted(edges, key=lambda e: e.sort_key):
        if e.sort_key in seen:
            continue
        seen.add(e.sort_key)
        unique.append(e)
    return unique


def build_counts(edges: list[Edge]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for e in edges:
        counts.setdefault(e.src, {})
        counts[e.src][e.dst] = counts[e.src].get(e.dst, 0) + 1
    return counts


def find_cycles(counts: dict[str, dict[str, int]]) -> list[list[str]]:
    """Return every elementary cycle in the top-level package graph."""
    cycles: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    def dfs(start: str, node: str, path: list[str]) -> None:
        for nxt in sorted(counts.get(node, {})):
            if nxt == start:
                key = frozenset(path)
                if len(path) > 1 and key not in seen:
                    seen.add(key)
                    cycles.append([*path, start])
            elif nxt not in path and len(path) < len(KNOWN_PACKAGES):
                dfs(start, nxt, [*path, nxt])

    for node in sorted(counts):
        dfs(node, node, [node])
    return cycles


def within(module: str, package: str) -> bool:
    """True if ``module`` is ``package`` itself or lives inside it.

    ``within("aios.core.models", "aios.core") -> True``
    ``within("aios.corex", "aios.core")    -> False``
    """
    return module == package or module.startswith(package + ".")


def rule_covers(rule: Rule, edge: Edge) -> bool:
    """True if ``rule`` is *about* this edge (whether it permits or forbids it).

    * ``allow`` -- governs *every* outgoing edge of the subject package, so
      an unexpected dependency is reported.
    * ``forbid`` on a top-level package -- governs that package and its
      sub-packages, and covers only the dependencies it actually forbids.
    * ``forbid``/``pair_forbid`` on a sub-package -- governs only edges
      originating inside that sub-package, and again only the listed targets.
    """
    if not within(edge.src_module, rule.subject):
        return False
    if rule.kind == "allow":
        return True
    return any(within(edge.dst_module, t) for t in rule.targets)


def rule_allows(rule: Rule, edge: Edge) -> bool:
    """True if ``rule`` permits this edge."""
    if rule.kind == "allow":
        # imports within the subject package itself are always fine
        return (
            within(edge.dst_module, rule.subject)
            or any(within(edge.dst_module, t) for t in rule.targets)
            or any(within(edge.dst_module, t) for t in rule.exempt)
        )
    # forbid / pair_forbid: anything not explicitly listed is permitted
    return not any(within(edge.dst_module, t) for t in rule.targets)


def check_rules(edges: list[Edge]) -> list[Violation]:
    """Apply every rule in :data:`RULES` to the resolved edges."""
    violations: list[Violation] = []
    for edge in edges:
        for rule in RULES:
            if not rule_covers(rule, edge) or rule_allows(rule, edge):
                continue
            violations.append(
                Violation(
                    rule=f"{rule.kind}:{rule.subject}",
                    src=edge.src,
                    dst=edge.dst,
                    file=edge.file,
                    line=edge.line,
                    reason=rule.reason,
                )
            )
    violations.sort(key=lambda v: (v.src, v.dst, v.file, v.line))
    return violations


def unruled_edges(edges: list[Edge]) -> list[Edge]:
    """Edges not covered by any allow/forbid rule (reported separately)."""
    out: list[Edge] = []
    for e in edges:
        if not any(rule_covers(rule, e) for rule in RULES):
            out.append(e)
    return out


def render_table(counts: dict[str, dict[str, int]]) -> str:
    packages = sorted(set(counts) | {p for v in counts.values() for p in v})
    rows: list[tuple[str, ...]] = [("importer (aios.*)", "imports", "count")]
    for src in packages:
        targets = counts.get(src, {})
        if not targets:
            rows.append((src, "-", "0"))
            continue
        for dst in sorted(targets, key=lambda d: (-targets[d], d)):
            rows.append((src, dst, str(targets[dst])))
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    lines = []
    for r in rows:
        lines.append(f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:>5}")
    return "\n".join(lines)


def render_mermaid(counts: dict[str, dict[str, int]]) -> str:
    packages = sorted(set(counts) | {p for v in counts.values() for p in v})
    ids = {p: p.replace(".", "_") for p in packages}
    lines = ["graph LR"]
    for p in packages:
        lines.append(f'  {ids[p]}["{p}"]')
    for src in packages:
        targets = counts.get(src, {})
        for dst, n in sorted(targets.items(), key=lambda kv: (-kv[1], kv[0])):
            label = f" x{n}" if n > 1 else ""
            lines.append(f"  {ids[src]} -->|{label}| {ids[dst]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print the graph as JSON")
    ap.add_argument(
        "--mermaid", action="store_true", help="print a Mermaid graph LR diagram"
    )
    ap.add_argument(
        "--strict", action="store_true", help="exit non-zero on any violation"
    )
    args = ap.parse_args(argv)

    if not AIOS_DIR.is_dir():
        print(f"error: {AIOS_DIR} not found", file=sys.stderr)
        return 2

    all_edges = build_graph(AIOS_DIR)
    # Intra-package imports (e.g. aios.core -> aios.core.models) are still fed
    # to the rule engine, but they would swamp the cross-package table.
    edges = [e for e in all_edges if e.src != e.dst]
    intra = [e for e in all_edges if e.src == e.dst]
    counts = build_counts(edges)
    violations = check_rules(all_edges)
    cycles = find_cycles(counts)
    unruled = unruled_edges(edges)

    if args.json:
        print(json.dumps({
            "nodes": sorted(set(counts) | {p for v in counts.values() for p in v}),
            "edges": [
                {"src": s, "dst": d, "count": n}
                for s in sorted(counts)
                for d, n in sorted(counts[s].items())
            ],
            "intra_package_imports": len(intra),
            "violations": [v.__dict__ for v in violations],
            "cycles": cycles,
            "unruled_edges": [
                {"src": e.src, "dst": e.dst, "file": e.file, "line": e.line}
                for e in unruled
            ],
        }, indent=2))
    elif args.mermaid:
        print(render_mermaid(counts))
    else:
        print("=" * 70)
        print("PACKAGE DEPENDENCY TABLE (importer -> imported : import statements)")
        print("=" * 70)
        print(render_table(counts))
        print(
            f"\n({len(intra)} intra-package imports within aios.* "
            "excluded from the table)"
        )
        print()
        print(f"LAYERING VIOLATIONS: {len(violations)}")
        if violations:
            for v in violations:
                print(f"  {v.file}:{v.line}: {v.src} -> {v.dst}  [{v.rule}] {v.reason}")
        else:
            print("  (none)")
        print()
        print(f"CYCLES BETWEEN TOP-LEVEL PACKAGES: {len(cycles)}")
        if cycles:
            for c in cycles:
                print("  " + " -> ".join(c))
        else:
            print("  (none)")
        print()
        unruled_counts: dict[tuple[str, str], int] = {}
        for e in unruled:
            unruled_counts[(e.src, e.dst)] = unruled_counts.get((e.src, e.dst), 0) + 1
        print("OBSERVED BUT NOT COVERED BY A RULE")
        if unruled_counts:
            for (s, d), n in sorted(unruled_counts.items()):
                print(f"  {s} -> {d} ({n} import statement{'s' if n != 1 else ''})")
        else:
            print("  (none)")

    if args.strict and (violations or cycles):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())