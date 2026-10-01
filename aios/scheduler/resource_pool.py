"""
ResourcePool — standalone capacity accounting, separated from queue management.

This class owns the invariant: ``available + reserved == total`` for every
resource dimension, and nothing else. It does not know about queues,
scheduling strategies, or request objects. It answers one question:
given current resource availability, can this demand be satisfied?

The separation exists because the original ``ResourceScheduler`` conflated
two distinct concerns:

1. **Capacity accounting** — tracking total/available/reserved counters,
   admission decisions (ACCEPTED/WAITING/REJECTED), reserve/release.
2. **Queue management** — FIFO/RR/Priority ordering, preemption, blocking.

``ResourcePool`` handles (1). ``ResourceScheduler`` delegates to it for (1)
and handles (2) itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from aios.core.exceptions import ResourceError, ValidationError
from aios.core.ids import RequestID, TaskID
from aios.core.models.request import AgentRequest
from aios.core.models.task import Task


class DecisionType(str, Enum):
    """
    Admission decision outcomes.

    ACCEPTED: Fits within current available capacity — can run now.
    WAITING:  Exceeds current available but not total — can run later.
    REJECTED: Exceeds total capacity — can never run.
    """

    ACCEPTED = "accepted"
    WAITING = "waiting"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReservationID:
    """Unique identifier for a resource reservation."""

    value: UUID

    @classmethod
    def generate(cls) -> ReservationID:
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class AdmissionDecision:
    """Result of a capacity admission check."""

    decision: DecisionType
    reason: str
    target_id: RequestID | TaskID | None = None
    required_resources: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceState:
    """Immutable snapshot of resource allocations."""

    total: dict[str, float]
    available: dict[str, float]
    reserved: dict[str, float]
    active_reservations: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ResourcePool:
    """
    Capacity ledger for one resource type.

    Tracks total, available, and reserved capacity across an arbitrary
    number of named dimensions (e.g. ``{"slots": 4, "tokens_per_min": 100_000}``).

    Every mutation asserts the invariant:
        ``available[k] + reserved[k] == total[k]``  for all k.

    Public surface (matches spec):
        can_admit(item)    -> AdmissionDecision
        reserve(item)      -> ReservationID
        release(res_id)    -> None          (by ReservationID)
        release_by_task(t) -> None          (by TaskID — convenience)
        snapshot()         -> ResourceState
    """

    def __init__(
        self,
        total_capacity: dict[str, float] | float = 100.0,
    ) -> None:
        # Normalise scalar to single-dimension dict
        if isinstance(total_capacity, (int, float)):
            self._total: dict[str, float] = {"units": float(total_capacity)}
        elif isinstance(total_capacity, dict):
            self._total = {k: float(v) for k, v in total_capacity.items()}
        else:
            raise ValidationError(
                "total_capacity must be a float or a dict[str, float]."
            )

        for k, v in self._total.items():
            if v < 0:
                raise ValidationError(
                    f"total_capacity['{k}'] must be >= 0, got {v}."
                )

        self._available: dict[str, float] = dict(self._total)
        self._reserved: dict[str, float] = {k: 0.0 for k in self._total}

        # reservation_id  ->  demands dict
        self._reservations: dict[ReservationID, dict[str, float]] = {}
        # task_id  ->  reservation_id  (reverse index for task-keyed release)
        self._task_reservations: dict[TaskID, ReservationID] = {}

        self._assert_invariants()

    # ------------------------------------------------------------------
    # Invariant enforcement
    # ------------------------------------------------------------------

    def _assert_invariants(self) -> None:
        """Enforce: available + reserved == total for every dimension."""
        for key, total_val in self._total.items():
            avail = self._available.get(key, 0.0)
            res = self._reserved.get(key, 0.0)
            if round(avail + res, 6) != round(total_val, 6):
                raise ResourceError(
                    f"Invariant violated for '{key}': "
                    f"available ({avail}) + reserved ({res}) != total ({total_val})"
                )

    # ------------------------------------------------------------------
    # Demand extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_demands(
        item: Task | AgentRequest | dict[str, float],
    ) -> tuple[dict[str, float], TaskID | RequestID | None]:
        """
        Pull resource-requirement dict and an identifying ID from *item*.

        Convention: requirements live in ``item.metadata["resources"]``.
        If absent, a default demand of ``{"units": 1.0}`` is assumed so
        callers never need to special-case the no-metadata path.
        """
        if isinstance(item, dict):
            return {k: float(v) for k, v in item.items()}, None

        meta = getattr(item, "metadata", {})
        if isinstance(meta, dict) and "resources" in meta:
            res = meta["resources"]
            if isinstance(res, dict):
                demands = {k: float(v) for k, v in res.items()}
            else:
                demands = {"units": 1.0}
        else:
            demands = {"units": 1.0}

        target_id = getattr(item, "request_id", None) or getattr(
            item, "task_id", None
        )
        return demands, target_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_admit(
        self, item: Task | AgentRequest | dict[str, float]
    ) -> AdmissionDecision:
        """
        Check if *item* can be admitted now, later, or never.

        Returns an ``AdmissionDecision`` whose ``.decision`` field is one of
        ``DecisionType.ACCEPTED``, ``WAITING``, or ``REJECTED``.
        """
        demands, target_id = self._extract_demands(item)

        # 1. Exceeds TOTAL -> REJECTED (can never be satisfied)
        for key, req_val in demands.items():
            tot = self._total.get(key)
            if tot is None or req_val > tot:
                return AdmissionDecision(
                    decision=DecisionType.REJECTED,
                    reason=(
                        f"Requested {key}={req_val} exceeds total capacity "
                        f"{tot if tot is not None else 0.0}."
                    ),
                    target_id=target_id,
                    required_resources=demands,
                )

        # 2. Exceeds AVAILABLE -> WAITING (can be satisfied later)
        for key, req_val in demands.items():
            avail = self._available.get(key, 0.0)
            if req_val > avail:
                return AdmissionDecision(
                    decision=DecisionType.WAITING,
                    reason=(
                        f"Requested {key}={req_val} exceeds available "
                        f"{avail} (total={self._total.get(key)})."
                    ),
                    target_id=target_id,
                    required_resources=demands,
                )

        # 3. Fits -> ACCEPTED
        return AdmissionDecision(
            decision=DecisionType.ACCEPTED,
            reason="Required resources are currently available.",
            target_id=target_id,
            required_resources=demands,
        )

    def reserve(
        self, item: Task | AgentRequest | dict[str, float]
    ) -> ReservationID:
        """
        Reserve capacity for an admitted item.

        Raises ``ResourceError`` if *item* cannot currently be admitted.
        """
        decision = self.can_admit(item)
        if decision.decision != DecisionType.ACCEPTED:
            raise ResourceError(f"Cannot reserve: {decision.reason}")

        demands = decision.required_resources
        reservation_id = ReservationID.generate()

        for key, req_val in demands.items():
            self._available[key] -= req_val
            self._reserved[key] += req_val

        self._reservations[reservation_id] = demands

        # Maintain the task-id reverse index when possible
        task_id = getattr(item, "task_id", None)
        if task_id is not None:
            self._task_reservations[task_id] = reservation_id

        self._assert_invariants()
        return reservation_id

    def release(self, reservation_id: ReservationID) -> None:
        """
        Release previously reserved capacity by ``ReservationID``.

        Raises ``ResourceError`` if the id is unknown or already released.
        """
        if (
            not isinstance(reservation_id, ReservationID)
            or reservation_id not in self._reservations
        ):
            raise ResourceError(
                f"Unknown or already released reservation: {reservation_id}"
            )

        demands = self._reservations.pop(reservation_id)
        for key, req_val in demands.items():
            self._available[key] += req_val
            self._reserved[key] -= req_val

        # Clean the reverse index
        to_remove = [
            tid
            for tid, rid in self._task_reservations.items()
            if rid == reservation_id
        ]
        for tid in to_remove:
            del self._task_reservations[tid]

        self._assert_invariants()

    def release_by_task(self, task_id: TaskID) -> None:
        """
        Release capacity by ``TaskID`` instead of ``ReservationID``.

        Convenience method so callers who only hold a task id don't need
        to track the reservation id separately.

        Raises ``ResourceError`` if no reservation exists for *task_id*.
        """
        reservation_id = self._task_reservations.get(task_id)
        if reservation_id is None:
            raise ResourceError(
                f"No reservation found for task {task_id}."
            )
        self.release(reservation_id)

    def snapshot(self) -> ResourceState:
        """Return an immutable snapshot of the current capacity state."""
        return ResourceState(
            total=dict(self._total),
            available=dict(self._available),
            reserved=dict(self._reserved),
            active_reservations=len(self._reservations),
        )
