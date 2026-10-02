# AIOS system-call message contract

Every kernel request uses the shared `AgentRequest` envelope:
`request_id`, `agent_id`, `task_id`, `syscall`, and optional `metadata`.
The `syscall` contains a `call_type` and a plain-object `payload`. Do not
replace this envelope with LLM-, tool-, memory-, or storage-specific request
classes. A response uses the shared `AgentResponse` envelope and repeats the
request's `request_id`; that `RequestID` is also the IPC correlation ID.

All payloads and response results must contain JSON-safe values: null,
booleans, finite numbers, strings, arrays, and objects with string keys.

## Syscall payloads

| Syscall | Required payload keys | Contract |
| --- | --- | --- |
| `LLM_CALL` | `model`, `messages`, `parameters` | `model` is a non-empty string. `messages` is a non-empty array of objects with string `role` and `content`; role is `system`, `user`, or `assistant`. `parameters` is an object. Optional `temperature` is numeric in `[0, 2]`, `max_tokens` is a positive integer, and `top_p` is numeric in `[0, 1]`. |
| `TOOL_CALL` | `tool_name` | Non-empty tool name; optional `arguments` is an object (use `{}` for a tool with no arguments). |
| `MEMORY_READ` | `resource_id` | Resource ID is a non-empty string or integer. |
| `MEMORY_WRITE` | `resource_id`, `value` | Resource ID is a non-empty string or integer; `value` may be any JSON-safe value. |
| `STORAGE_READ` | `resource_id` | Resource ID is a non-empty string or integer. |
| `STORAGE_WRITE` | `resource_id`, `value` | Resource ID is a non-empty string or integer; `value` may be any JSON-safe value. |

## Responses

A successful `AgentResponse` has `status="success"`, a non-null JSON-safe
`result`, and no `error`. A failed response has `status="failed"`, a non-empty
`error` string, and no `result`. `metadata` is an object containing only
JSON-safe values.

## JSON envelope serialization

`aios.communication.protocol.request_to_dict` and `request_from_dict`
serialize/restore the kernel request wire shape (`request_id`, `agent_id`,
`task_id`, `call_type`, `payload`, `metadata`). As in
`aios.kernel.serialization`, optional `context` is not serialized because it
does not yet have a published wire representation. The response pair uses
`request_id`, `status`, `result`, `error`, and `metadata`.

Validation is independent of transport: `IPCChannel` carries messages and
correlates them by `RequestID`; the protocol module defines whether envelopes
and payloads are well formed.