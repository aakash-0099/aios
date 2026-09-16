# AIOS — Agent Operating System

AIOS is an agent operating/runtime layer built on top of a normal operating system (Windows, Linux, macOS). It is not a replacement for the OS — it sits above it, giving AI agents the same kind of structured, managed access to resources that a traditional OS gives to processes: identity, scheduling, isolated memory and storage, controlled tool access, and a stable system-call boundary.

The goal is a **domain-neutral platform**: any agent, for any application, can be attached through a stable SDK and get the same lifecycle, scheduling, context, memory, storage, LLM, and tool/security capabilities — without the kernel needing to know anything about that agent's specific purpose.

## Core Idea

```
Agent → AIOS SDK → System Call → AIOS Kernel → Managed Resource
```

An agent never touches a resource (an LLM provider, a memory store, the filesystem, an external tool) directly. It goes through the SDK, which issues a **system call**, which the **kernel** validates, authenticates, and dispatches to the correct manager. This mirrors how a normal OS mediates access to hardware and services — except the "processes" here are AI agents, and the "hardware" is memory, storage, tools, and language models.

## System Calls

AIOS defines exactly six system calls at the kernel boundary:

| Syscall | Purpose |
|---|---|
| `LLM_CALL` | Invoke a language model through a provider-independent interface |
| `MEMORY_READ` / `MEMORY_WRITE` | Agent-scoped memory access |
| `STORAGE_READ` / `STORAGE_WRITE` | Durable, agent-scoped storage access |
| `TOOL_CALL` | Controlled, authorized invocation of external tools |

Every syscall has a fixed request/response shape, a defined owner, and defined error behavior. This contract is frozen early so that different components can be built independently and in parallel against it.

## Architectural Components

| Component | Responsibility |
|---|---|
| **Kernel** | Central dispatcher — validates requests, routes syscalls to handlers, normalizes errors and responses |
| **Agent Manager** | Owns agent identity and lifecycle (create, pause, resume, terminate) |
| **Scheduler** | Selects which agent's task runs next; handles priority and fairness |
| **Context Manager** | Owns per-agent execution state; saves/restores context across agent switches |
| **Memory Manager** | Agent-scoped, isolated memory read/write |
| **Storage Manager** | Agent-scoped, durable storage read/write |
| **Tool Manager & Access Control** | Deny-by-default, audited execution of external tools |
| **LLM Core** | Provider-independent abstraction over language model backends |
| **SDK** | The only interface an agent developer needs — hides all kernel internals |

## Key Invariants

These hold throughout the system, not just in individual components:

- **Agent isolation is mandatory.** One agent can never read, overwrite, or inherit another agent's context, memory, storage, or permissions.
- **Agent identity is explicit** in every request and attributable in every response.
- **Every syscall is validated before dispatch**, and unauthorized operations fail deterministically.
- **All errors flow through a shared exception hierarchy** — no component invents its own error types.
- **Interfaces are contracts, not conventions.** A component depends on another team's *published* interface, never on its internal implementation. Mocks and contract tests are treated as permanent development artifacts, not throwaway scaffolding.
- **The kernel stays domain-neutral.** Nothing application-specific belongs below the SDK layer.

## Repository Layout

```
aios/            Core implementation — kernel, models, scheduler, context,
                 memory, storage, tools, security, LLM, SDK
agents/          Example and domain-neutral agent programs
tools/           Tool definitions/adapters usable by agents
tests/           Unit, contract, integration, isolation, concurrency, e2e tests
evaluation/      Benchmarks, baselines, experiment runners, results
docs/            Architecture, contracts, decisions, runbooks
scripts/         Developer and evaluation utilities
data/            Local fixtures and sample resources (no secrets)
```

## What Gets Built, and in What Order

The system is built in three broad stages:

**1. Foundation (sequential).** Shared data types (`Agent`, `Task`, `AgentRequest`, `AgentResponse`, etc.), the exception hierarchy, and the kernel dispatcher with mock handlers for all six syscalls. Nothing downstream starts until this contract is frozen.

**2. Parallel build (independent workstreams).** With the syscall contract frozen, these can be built simultaneously by separate teams, each replacing one kernel mock:
- LLM Core (`LLM_CALL`)
- Agent Manager & lifecycle
- Scheduler
- Context Manager
- Memory Manager (`MEMORY_READ` / `MEMORY_WRITE`)
- Storage Manager (`STORAGE_READ` / `STORAGE_WRITE`)
- Tool Manager & Access Control (`TOOL_CALL`)

Each module is unit-tested and contract-tested in isolation, against mocks of everything else.

**3. Integration and evaluation (sequential).** Real modules replace the mocks behind the frozen kernel contract. The SDK is built on top, exposing `agent.llm`, `agent.memory`, `agent.storage`, and `agent.tools` as the stable developer-facing API. Finally, a multi-agent runtime is exercised at increasing concurrency (1, 5, 10, 20 agents) to prove isolation and absence of cross-agent leakage, and the whole system is benchmarked against a non-AIOS baseline.

## Design Philosophy

- A component built against a syscall contract should never need to change when the thing behind that contract changes from a mock to a real implementation.
- Isolation is treated as a correctness property to be tested under concurrency, not an assumption.
- Tool and resource access is deny-by-default and always attributable to a specific agent and request.
- The platform is judged successful when a *new* agent can be plugged in through the SDK and get full OS-level capabilities — lifecycle, scheduling, context, memory, storage, LLM access, tool access — without anyone touching the kernel.
