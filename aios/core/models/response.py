"""
Agent response model.

AgentResponse is the canonical response returned by AIOS for
an AgentRequest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import RequestID


class RequestStatus(str, Enum):
    """
    Final status of an AIOS request.
    """

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentResponse:
    """
    Response corresponding to an AgentRequest.

    A successful response contains a result.

    A failed response contains an error description.
    """

    request_id: RequestID
    status: RequestStatus

    result: Any = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate response semantics.
        """

        if not isinstance(self.request_id, RequestID):
            raise ValidationError(
                "request_id must be a RequestID."
            )

        if not isinstance(self.status, RequestStatus):
            raise ValidationError(
                "status must be a RequestStatus."
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        # A successful request must provide a result.
        if self.status == RequestStatus.SUCCESS:
            if self.result is None:
                raise ValidationError(
                    "Successful responses must contain a result."
                )

            if self.error is not None:
                raise ValidationError(
                    "Successful responses cannot contain an error."
                )

        # A failed request must provide an error.
        if self.status == RequestStatus.FAILED:
            if self.error is None or not self.error.strip():
                raise ValidationError(
                    "Failed responses must contain an error."
                )

            if self.result is not None:
                raise ValidationError(
                    "Failed responses cannot contain a result."
                )