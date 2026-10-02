# Development setup

Python **3.11** is required (matches `requires-python` in `pyproject.toml`).

## Create the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

There is no separate dev-requirements file: `requirements.txt` already
includes the test and lint tooling (pytest, pytest-asyncio, pytest-cov,
ruff, black, mypy). The venv is git-ignored -- never commit it.

Verify (both should succeed):

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import pydantic, yaml, requests, pytest_asyncio, redis, openai, groq"
```

## Run the tests

Plain `pytest` from the repo root runs the whole suite. Config lives in
`[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths`,
`asyncio_mode = "auto"`, `norecursedirs`, `markers`).

```powershell
.\.venv\Scripts\python.exe -m pytest                   # whole suite
.\.venv\Scripts\python.exe -m pytest -q -rs            # + skip reasons
.\.venv\Scripts\python.exe -m pytest tests\unit        # subset
.\.venv\Scripts\python.exe -m pytest --collect-only -q # list node IDs
```

Expected: **546 passed, 2 skipped**. Both skips are the symlink escape tests
in `aios/storage/tests/`, which are Windows-specific and expected to skip.
`aios/llm/tests/` only re-exports tests from `tests/unit/` and
`tests/integration/`, so it sits outside `testpaths` on purpose -- including
it would collect those tests twice.

## Run the Redis tests

The Redis-backed syscall queue (`aios/kernel/queue_backend.py`) is optional;
`tests/unit/test_redis_queue_backend.py` skips itself when no server is
reachable. Use Docker rather than a system-wide install:

```powershell
docker run --rm -d -p 6379:6379 --name aios-redis redis:7
.\.venv\Scripts\python.exe -m pytest tests\unit\test_redis_queue_backend.py -v
docker stop aios-redis
```

For another port/server set `REDIS_URL` (default `redis://localhost:6379/0`):

```powershell
$env:REDIS_URL = "redis://localhost:6380/0"
.\.venv\Scripts\python.exe -m pytest tests\unit\test_redis_queue_backend.py -v
Remove-Item Env:\REDIS_URL
```

## Run the import / layering checker

```powershell
.\.venv\Scripts\python.exe scripts\check_imports.py
```

Prints the package-to-package dependency table, rule violations with
`file:line`, dependency cycles, and edges not covered by a rule. Edit the
`RULES` list at the top of that file to change the policy.

| Flag | Effect |
| --- | --- |
| *(none)* | human-readable report, always exits 0 |
| `--strict` | exit 1 on any violation or cycle (for CI) |
| `--json` | graph, violations and unruled edges as JSON |
| `--mermaid` | Mermaid `graph LR` diagram |

To rebuild from scratch, delete `.venv` and repeat the two commands above.