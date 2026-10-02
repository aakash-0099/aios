"""Validation and JSON-safe serialization for AIOS request/response envelopes.

The protocol validates the canonical ``AgentRequest`` and ``AgentResponse``
models. It does not define parallel envelopes or perform IPC transport.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any
from uuid import UUID

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID, RequestID, TaskID
from aios.core.models import (
    AgentRequest,
    AgentResponse,
    RequestStatus,
    SystemCall,
    SystemCallType,
)


def _is_json_value(value: Any) -> bool:
    """Return whether a value can be represented by strict JSON."""
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _require_json_value(value: Any, name: str) -> None:
    if not _is_json_value(value):
        raise ValidationError(f"{name} must contain only JSON-safe values.")


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string.")


def _validate_llm_payload(payload: dict[str, Any]) -> None:
    required = {"model", "messages", "parameters"}
    missing = required - payload.keys()
    if missing:
        raise ValidationError(
            f"LLM_CALL payload is missing required keys: {sorted(missing)}."
        )
    _require_non_empty_string(payload["model"], "LLM_CALL model")

    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValidationError("LLM_CALL messages must be a non-empty list.")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"LLM_CALL messages[{index}] must be an object.")
        if any(not isinstance(key, str) for key in message):
            raise ValidationError(
                f"LLM_CALL messages[{index}] keys must be strings."
            )
        if any(not isinstance(value, str) for value in message.values()):
            raise ValidationError(
                f"LLM_CALL messages[{index}] values must be strings."
            )
        if not {"role", "content"} <= message.keys():
            raise ValidationError(
                f"LLM_CALL messages[{index}] must contain role and content."
            )
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValidationError(
                f"LLM_CALL messages[{index}] has an invalid role."
            )

    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValidationError("LLM_CALL parameters must be an object.")
    _require_json_value(parameters, "LLM_CALL parameters")

    if "temperature" in parameters:
        temperature = parameters["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= temperature <= 2.0
        ):
            raise ValidationError(
                "LLM_CALL temperature must be a number between 0.0 and 2.0."
            )
    if "max_tokens" in parameters:
        max_tokens = parameters["max_tokens"]
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValidationError("LLM_CALL max_tokens must be a positive integer.")
    if "top_p" in parameters:
        top_p = parameters["top_p"]
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0.0 <= top_p <= 1.0
        ):
            raise ValidationError(
                "LLM_CALL top_p must be a number between 0.0 and 1.0."
            )


def _validate_resource_payload(
    payload: dict[str, Any],
    operation: str,
    *,
    write: bool,
) -> None:
    if "resource_id" not in payload:
        raise ValidationError(f"{operation} payload requires resource_id.")
    resource_id = payload["resource_id"]
    if (
        resource_id is None
        or isinstance(resource_id, bool)
        or not isinstance(resource_id, (str, int))
        or (isinstance(resource_id, str) and not resource_id.strip())
    ):
        raise ValidationError(
            f"{operation} resource_id must be a non-empty string or integer."
        )
    if write and "value" not in payload:
        raise ValidationError(f"{operation} payload requires value.")


def _validate_tool_payload(payload: dict[str, Any]) -> None:
    _require_non_empty_string(payload.get("tool_name"), "TOOL_CALL tool_name")
    if "arguments" in payload and not isinstance(payload["arguments"], dict):
        raise ValidationError("TOOL_CALL arguments must be an object.")
    _require_json_value(payload.get("arguments", {}), "TOOL_CALL arguments")


def validate_request(agent_request: AgentRequest) -> None:
    """Validate an ``AgentRequest`` and the payload for its declared syscall.

    Payload contract:
    - ``LLM_CALL``: ``model``, ``messages`` (role/content objects), and
      ``parameters``; key names and known generation limits match ``LLMPayload``.
    - ``TOOL_CALL``: non-empty ``tool_name`` and optional object ``arguments``.
    - ``MEMORY_READ``/``STORAGE_READ``: ``resource_id``.
    - ``MEMORY_WRITE``/``STORAGE_WRITE``: ``resource_id`` and ``value``.
    """
    if not isinstance(agent_request, AgentRequest):
        raise ValidationError("agent_request must be an AgentRequest.")

    syscall = agent_request.syscall
    if not isinstance(syscall.call_type, SystemCallType):
        raise ValidationError("request syscall call_type is invalid.")
    payload = syscall.payload
    if not isinstance(payload, dict):
        raise ValidationError("request syscall payload must be an object.")
    _require_json_value(payload, "request syscall payload")

    call_type = syscall.call_type
    if call_type == SystemCallType.LLM_CALL:
        _validate_llm_payload(payload)
    elif call_type == SystemCallType.TOOL_CALL:
        _validate_tool_payload(payload)
    elif call_type == SystemCallType.MEMORY_READ:
        _validate_resource_payload(payload, "MEMORY_READ", write=False)
    elif call_type == SystemCallType.MEMORY_WRITE:
        _validate_resource_payload(payload, "MEMORY_WRITE", write=True)
    elif call_type == SystemCallType.STORAGE_READ:
        _validate_resource_payload(payload, "STORAGE_READ", write=False)
    elif call_type == SystemCallType.STORAGE_WRITE:
        _validate_resource_payload(payload, "STORAGE_WRITE", write=True)

    _require_json_value(agent_request.metadata, "request metadata")


def validate_response(agent_response: AgentResponse) -> None:
    """Validate response status/result/error consistency and JSON safety."""
    if not isinstance(agent_response, AgentResponse):
        raise ValidationError("agent_response must be an AgentResponse.")
    if not isinstance(agent_response.status, RequestStatus):
        raise ValidationError("response status is invalid.")
    _require_json_value(agent_response.metadata, "response metadata")

    if agent_response.status == RequestStatus.SUCCESS:
        if agent_response.result is None:
            raise ValidationError("Successful responses must contain a result.")
        if agent_response.error is not None:
            raise ValidationError("Successful responses cannot contain an error.")
        _require_json_value(agent_response.result, "response result")
    else:
        if (
            not isinstance(agent_response.error, str)
            or not agent_response.error.strip()
        ):
            raise ValidationError("Failed responses must contain a non-empty error.")
        if agent_response.result is not None:
            raise ValidationError("Failed responses cannot contain a result.")


def request_to_dict(agent_request: AgentRequest) -> dict[str, Any]:
    """Serialize an ``AgentRequest`` to the kernel envelope's JSON-safe shape.

    Like ``aios.kernel.serialization.request_to_dict``, this intentionally
    excludes the optional context, which has no published wire representation.
    """
    validate_request(agent_request)
    return {
        "request_id": str(agent_request.request_id),
        "agent_id": str(agent_request.agent_id),
        "task_id": str(agent_request.task_id),
        "call_type": agent_request.syscall.call_type.value,
        "payload": deepcopy(agent_request.syscall.payload),
        "metadata": deepcopy(agent_request.metadata),
    }


def request_from_dict(data: dict[str, Any]) -> AgentRequest:
    """Deserialize a dict produced by :func:`request_to_dict`."""
    if not isinstance(data, dict):
        raise ValidationError("Serialized request must be an object.")
    try:
        request = AgentRequest(
            request_id=RequestID(UUID(data["request_id"])),
            agent_id=AgentID(UUID(data["agent_id"])),
            task_id=TaskID(UUID(data["task_id"])),
            syscall=SystemCall(
                call_type=SystemCallType(data["call_type"]),
                payload=data.get("payload", {}),
            ),
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise ValidationError(
            f"Serialized request is missing field: {exc}."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Serialized request is malformed: {exc}.") from exc
    validate_request(request)
    return request


def response_to_dict(agent_response: AgentResponse) -> dict[str, Any]:
    """Serialize an ``AgentResponse`` into a JSON-safe dictionary."""
    validate_response(agent_response)
    return {
        "request_id": str(agent_response.request_id),
        "status": agent_response.status.value,
        "result": deepcopy(agent_response.result),
        "error": agent_response.error,
        "metadata": deepcopy(agent_response.metadata),
    }


def response_from_dict(data: dict[str, Any]) -> AgentResponse:
    """Deserialize a dict produced by :func:`response_to_dict`."""
    if not isinstance(data, dict):
        raise ValidationError("Serialized response must be an object.")
    try:
        response = AgentResponse(
            request_id=RequestID(UUID(data["request_id"])),
            status=RequestStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise ValidationError(
            f"Serialized response is missing field: {exc}."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Serialized response is malformed: {exc}.") from exc
    validate_response(response)
    return response