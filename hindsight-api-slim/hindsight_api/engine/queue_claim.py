"""Task-local fencing helpers for worker-owned queue claim generations.

The worker binds an exact ``status + worker_id + claim_token`` generation at
``MemoryEngine.execute_task``.  Retain code runs several modules below that
entrypoint, so the context and SQL helpers live here instead of in
``memory_engine`` to avoid circular imports.
"""

import contextvars
import functools
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from ..worker.exceptions import OperationQueueAuthorityError


@dataclass(frozen=True)
class QueueClaim:
    """Exact worker claim generation bound to one task execution."""

    worker_id: str | None
    claim_token: str | None


@dataclass(frozen=True)
class QueueClaimPredicate:
    """SQL fragment and parameters for an exact queue claim fence."""

    sql: str
    args: tuple[str, ...]


_current_queue_claim: contextvars.ContextVar[QueueClaim | None] = contextvars.ContextVar(
    "current_queue_claim", default=None
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def bind_queue_claim(
    arg: str = "task_dict",
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Bind private worker claim fields for the wrapped coroutine's duration."""

    def decorate(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            value = sig.bind(*args, **kwargs).arguments.get(arg)
            claim: QueueClaim | None = None
            if isinstance(value, dict):
                worker_id = value.get("_worker_id")
                claim_token = value.get("_claim_token")
                if worker_id is not None or claim_token is not None:
                    claim = QueueClaim(
                        worker_id=worker_id if isinstance(worker_id, str) else None,
                        claim_token=claim_token if isinstance(claim_token, str) else None,
                    )
            token = _current_queue_claim.set(claim)
            try:
                return await func(*args, **kwargs)
            finally:
                _current_queue_claim.reset(token)

        return wrapper

    return decorate


def active_queue_claim() -> QueueClaim | None:
    """Return a complete claim or fail closed on malformed worker context."""

    claim = _current_queue_claim.get()
    if claim is not None and (not claim.worker_id or not claim.claim_token):
        raise OperationQueueAuthorityError("Worker task is missing complete claim-generation authority")
    return claim


def queue_claim_predicate(first_parameter: int) -> QueueClaimPredicate:
    """Build an exact status/owner/generation SQL fence for worker reads or writes."""

    claim = active_queue_claim()
    if claim is None:
        return QueueClaimPredicate(sql="", args=())
    assert claim.worker_id is not None and claim.claim_token is not None
    return QueueClaimPredicate(
        sql=(f" AND status = 'processing' AND worker_id = ${first_parameter} AND claim_token = ${first_parameter + 1}"),
        args=(claim.worker_id, claim.claim_token),
    )


def _command_row_count(result: str | None) -> int:
    if not result:
        return 0
    try:
        return int(result.rsplit(maxsplit=1)[-1])
    except (TypeError, ValueError):
        return 0


def require_guarded_update(result: str | None, operation_id: str, action: str) -> None:
    """Reject a zero-row worker write without touching a successor claim."""

    if active_queue_claim() is not None and _command_row_count(result) != 1:
        raise OperationQueueAuthorityError(
            f"Lost queue claim generation while attempting to {action} operation {operation_id}"
        )


def require_guarded_row(row: Any, operation_id: str, action: str) -> None:
    """Reject a missing worker read that was fenced to the bound generation."""

    if active_queue_claim() is not None and row is None:
        raise OperationQueueAuthorityError(
            f"Lost queue claim generation while attempting to {action} operation {operation_id}"
        )
