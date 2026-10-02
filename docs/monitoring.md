# Monitoring / tracing

`aios.monitoring` is the tracing foundation. The default sink is a
no-op: with tracing unconfigured the system behaves exactly as before
and the overhead is negligible. `emit()` never raises and is
thread-safe, so instrumenting a component can never change its
behaviour.

## Event schema

Every event is an immutable `TraceEvent` (see `aios/monitoring/events.py`):

| Field | Meaning |
| --- | --- |
| `wall_time` | epoch seconds (`time.time()`), for correlation with logs |
| `mono_time` | `time.monotonic()`, for gap-free duration maths |
| `correlation_id` | ties one request's events together (the AIOS `RequestID`) |
| `agent_id` | the requesting agent |
| `component` | emitting package, e.g. `"kernel"` |
| `event` | event name, see the convention below |
| `data` | event-specific JSON-safe key/value pairs |

## Enabling tracing

```python
from aios.monitoring import JsonlSink, configure

configure(JsonlSink("traces/demo.jsonl"))   # one JSON object per line
```

For tests and demos, capture events in memory instead:

```python
from aios.monitoring import capture

with capture() as sink:
    ...                                    # run the system
    sink.events()                          # list[TraceEvent]
```

Render any trace file as text:

```powershell
.\.venv\Scripts\python.exe -m aios.monitoring.render traces\demo.jsonl
```

## Adding emit() calls in a new package

1. Add your event names to `aios/monitoring/events.py` in a new
   class (e.g. `class SchedulerEvents`), following the convention.
2. Import and call `emit` at the lifecycle points you want to
   observe. Always pass `correlation_id` and `agent_id` when
   available. Example for a scheduler:

```python
from aios.monitoring import emit
from aios.monitoring.events import SchedulerEvents

emit(SchedulerEvents.TASK_QUEUED,
     correlation_id=str(request.request_id),
     agent_id=str(request.agent_id),
     task_id=str(request.task_id))
```

Non-JSON-serialisable values in `**data` are coerced with `repr()`,
so `emit` can never fail on bad data.

## Naming convention

Event names are `<noun>.<past_tense_verb>`, namespaced by the noun
they describe: `syscall.created`, `syscall.queued`,
`syscall.handler_invoked`, `syscall.completed`, `syscall.failed`,
`response.returned`. The `component` field carries the package name,
so the same noun namespace can be reused across packages.

## Running the demo

```powershell
.\.venv\Scripts\python.exe scripts\demo_trace.py --runs 3 --agents 3
```

Runs 3 agents x 3 syscalls concurrently through the Kernel, writes
`traces/demo.jsonl` (git-ignored), and prints the timeline view
(events grouped by correlation id, with gaps) and the lanes view
(one lane per agent).
