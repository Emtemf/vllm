# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, process-local TTFT diagnostics; never a source of scheduling state."""

import contextvars
import itertools
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Any

_ENABLED = os.environ.get("VLLM_TTFT_DIAGNOSTICS", "0") == "1"
_LOGGER = logging.getLogger("vllm.ttft_diagnostics")
_HOST = socket.gethostname() if _ENABLED else ""
_RUN = uuid.uuid4().hex if _ENABLED else ""
_SEQUENCE = itertools.count()
_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "ttft_diagnostic_context", default=None
)


def enabled() -> bool:
    return _ENABLED


def new_id(kind: str) -> str:
    """Return a process/run-qualified identifier, not a cache or request key."""
    if not _ENABLED:
        return ""
    return f"{_RUN}:{os.getpid()}:{kind}:{next(_SEQUENCE)}"


def current() -> dict[str, Any]:
    """Copy diagnostic context before handing work to another thread."""
    return dict(_CONTEXT.get() or {}) if _ENABLED else {}


def emit(event: str, **fields: Any) -> None:
    if not _ENABLED:
        return
    import json

    record = dict(_CONTEXT.get() or {})
    record.update(fields)
    record.update(
        version=1,
        event=event,
        run_id=_RUN,
        host=_HOST,
        pid=os.getpid(),
        mono_ns=time.perf_counter_ns(),
        wall_ns=time.time_ns(),
    )
    _LOGGER.info(
        "[TTFT_DIAG] %s",
        json.dumps(record, ensure_ascii=True, separators=(",", ":")),
    )


@contextmanager
def scope(**fields: Any):
    """Scope metadata to this context; thread handoffs must copy it explicitly."""
    if not _ENABLED:
        yield
        return
    values = current()
    values.update(fields)
    token = _CONTEXT.set(values)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@contextmanager
def span(event: str, **fields: Any):
    """Emit paired boundaries without synchronizing or changing exceptions."""
    if not _ENABLED:
        yield None
        return
    parent_span_id = fields.get("span_id") or (_CONTEXT.get() or {}).get("span_id")
    if parent_span_id:
        fields["parent_span_id"] = parent_span_id
    span_id = new_id("span")
    fields["span_id"] = span_id
    with scope(**fields):
        start = time.perf_counter_ns()
        emit(event + "_start", start_ns=start)
        try:
            yield span_id
        except BaseException as exc:
            end = time.perf_counter_ns()
            emit(
                event,
                start_ns=start,
                end_ns=end,
                duration_ns=end - start,
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        else:
            end = time.perf_counter_ns()
            emit(
                event,
                start_ns=start,
                end_ns=end,
                duration_ns=end - start,
                status="ok",
            )
