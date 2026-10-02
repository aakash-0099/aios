"""
Synthetic tests for the layering rules in scripts/check_imports.py.

Each rule is proven to fail (and to pass) against a small synthetic
source tree built in tmp_path, so a rule can never silently become
a no-op.  The checker is loaded from scripts/ via importlib and
pointed at the synthetic tree by monkeypatching its ROOT/AIOS_DIR
module globals.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CHECK_IMPORTS_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_imports.py"
)

_spec = importlib.util.spec_from_file_location(
    "check_imports", CHECK_IMPORTS_PATH
)
assert _spec is not None and _spec.loader is not None
check_imports = importlib.util.module_from_spec(_spec)
# dataclasses resolves types through sys.modules at class-creation time
sys.modules["check_imports"] = check_imports
_spec.loader.exec_module(check_imports)


def _violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
) -> list[check_imports.Violation]:
    """
    Materialize `files` (repo-relative paths) under tmp_path, point
    the checker at the synthetic tree, and return the violations.
    """

    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(check_imports, "ROOT", tmp_path)
    monkeypatch.setattr(check_imports, "AIOS_DIR", tmp_path / "aios")

    edges = check_imports.build_graph(check_imports.AIOS_DIR)
    return check_imports.check_rules(edges)


def _rules(violations: list[check_imports.Violation]) -> list[str]:
    return [violation.rule for violation in violations]


# ---------------------------------------------------------------
# Existing rules keep working
# ---------------------------------------------------------------


def test_scheduler_still_forbidden_from_kernel(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/scheduler/scheduler.py": "from aios.kernel import Kernel\n"},
    )

    assert "forbid:aios.scheduler" in _rules(violations)


def test_agents_task_still_forbidden_from_agent_manager(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {
            "aios/agents/task.py": (
                "from aios.agents.agent_manager import AgentManager\n"
            ),
        },
    )

    assert "pair_forbid:aios.agents.task" in _rules(violations)


# ---------------------------------------------------------------
# aios.scheduler may import aios.agents ONLY via aios.agents.task
# ---------------------------------------------------------------


def test_scheduler_may_import_agents_task_module(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {
            "aios/scheduler/fifo.py": (
                "from aios.agents.task import validate_transition\n"
            ),
        },
    )

    assert violations == []


def test_scheduler_may_import_task_via_the_agents_package(tmp_path, monkeypatch):
    # `from aios.agents import task` resolves to aios.agents.task
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/scheduler/fifo.py": "from aios.agents import task\n"},
    )

    assert violations == []


def test_scheduler_cannot_import_agent_manager_via_the_package(
    tmp_path, monkeypatch
):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/scheduler/fifo.py": "from aios.agents import agent_manager\n"},
    )

    assert "allow_from:aios.scheduler" in _rules(violations)


def test_scheduler_cannot_import_other_agents_modules(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {
            "aios/scheduler/fifo.py": (
                "from aios.agents.agent_manager import AgentManager\n"
            ),
        },
    )

    assert "allow_from:aios.scheduler" in _rules(violations)


def test_scheduler_cannot_import_the_agents_package_itself(
    tmp_path, monkeypatch
):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/scheduler/fifo.py": "import aios.agents\n"},
    )

    assert "allow_from:aios.scheduler" in _rules(violations)


# ---------------------------------------------------------------
# aios.monitoring may import only aios.core
# ---------------------------------------------------------------


def test_monitoring_may_import_core(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/monitoring/tracer.py": "from aios.core.models import Task\n"},
    )

    assert violations == []


def test_monitoring_cannot_import_kernel(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/monitoring/tracer.py": "from aios.kernel import Kernel\n"},
    )

    assert "allow:aios.monitoring" in _rules(violations)


# ---------------------------------------------------------------
# Only aios.kernel imports aios.kernel (report-only)
# ---------------------------------------------------------------


def test_kernel_importing_kernel_is_allowed(tmp_path, monkeypatch):
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/kernel/kernel.py": "from aios.kernel.dispatcher import Dispatcher\n"},
    )

    assert violations == []


def test_other_packages_importing_kernel_are_reported_not_enforced(
    tmp_path, monkeypatch
):
    # aios.context has no rule of its own about aios.kernel, so the
    # only observation is the report-only forbid_unless rule.
    violations = _violations(
        tmp_path,
        monkeypatch,
        {"aios/context/manager.py": "from aios.kernel import Kernel\n"},
    )

    unless_violations = [
        v for v in violations if v.rule == "forbid_unless:*"
    ]
    assert len(unless_violations) == 1
    assert unless_violations[0].report_only is True

    # report-only: --strict must still exit 0
    assert check_imports.main(["--strict"]) == 0


def test_enforced_violation_fails_strict(tmp_path, monkeypatch):
    _violations(
        tmp_path,
        monkeypatch,
        {"aios/scheduler/fifo.py": "from aios.agents import agent_manager\n"},
    )

    assert check_imports.main(["--strict"]) == 1
