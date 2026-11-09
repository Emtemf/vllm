# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host diagnostics, always on; not an additive TTFT accounting model.

Emitted unconditionally (VLLM_TTFT_DIAGNOSTICS is no longer read; the
env gate was removed so logs are always available).
Monotonic timestamps are comparable only within a host clock domain; stages
and ranks can overlap. No device synchronization is performed, and logging
failures are non-fatal.

Scheduler payloads carry execution_id through queued results. Each worker emits
one untimed execution identity, including ranks that do not save KV, and scopes
that identity for downstream store capture. Deferred saves retain the scope of
their originating execution even though sample RPCs carry no execution ID.
Native backend-put and rendering/tokenization spans emit one completion with
monotonic bounds and start wall/thread provenance. Overlapping host spans must
not be summed as request time.
The separate QUEUE record is subtractable only when initial_queue_valid=true;
it never represents total queueing after requeue or preemption.
"""

import itertools
import logging
import os
import re
import socket
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

_LOGGER = logging.getLogger("vllm.ttft_diagnostics")
_RUN_ID = uuid.uuid4().hex
_HOST = socket.gethostname()
_COUNTER = itertools.count()
_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "ttft_diagnostics", default=None
)


def enabled() -> bool:
    """Diagnostics are always on; kept for call-site compatibility."""
    return True


def new_id(kind: str) -> str:
    """Return a bounded ASCII identifier, unique across forked processes too."""
    kind = "".join(
        c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in kind[:64]
    )
    return f"{kind}-{_RUN_ID}-{os.getpid()}-{next(_COUNTER)}"[:256]


def current() -> dict[str, Any]:
    return (_CONTEXT.get() or {}).copy()


# Greppable offline format: `[TTFT_DIAG] event=<name> key=value ...`.
# Values are single tokens; anything with space/=/quote/comma is "-quoted with
# \ and " escaped. None -> null (missing, never zero); bool -> true/false;
# empty sequence -> []; sequences -> comma-joined (no spaces).
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.:/+-]+$")


def _render_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        if not value:
            return "[]"
        items = (
            sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        )
        return ",".join(_render_value(v) for v in items)
    text = str(value)
    if _SAFE_VALUE.match(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return '"' + escaped + '"'


def _format_line(event: str, record: dict[str, Any]) -> str:
    parts = ["event=" + event]
    for key, value in record.items():
        if key == "event":
            continue
        parts.append(key + "=" + _render_value(value))
    return " ".join(parts)


def emit(event: str, **fields: Any) -> None:
    try:
        record = current()
        # A point event is not the enclosing span's end event. Explicit span
        # fields supplied by _span below are retained, inherited ones are not.
        record.pop("span_id", None)
        record.pop("parent_span_id", None)
        record.update(fields)
        record.update(
            version=1,
            event=event,
            host=_HOST,
            pid=os.getpid(),
            native_tid=threading.get_native_id(),
            run_id=_RUN_ID,
            mono_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
        )
        _LOGGER.info("[TTFT_DIAG] %s", _format_line(event, record))
    except Exception:
        # Diagnostic serialization/handlers must not replace business failures.
        pass


@contextmanager
def _scope(fields: dict[str, Any]):
    token = None
    try:
        context = current()
        context.update(fields)
        token = _CONTEXT.set(context)
    except Exception:
        pass
    try:
        yield
    finally:
        if token is not None:
            with suppress(Exception):
                _CONTEXT.reset(token)


def scope(**fields: Any):
    return _scope(fields)


@contextmanager
def _span(event: str, fields: dict[str, Any]):
    token = None
    start_ns = None
    try:
        context = current()
        parent_span_id = context.get("span_id")
        span_id = new_id("span")
        start_ns = time.monotonic_ns()
        fields.update(
            span_id=span_id,
            start_ns=start_ns,
            start_wall_ns=time.time_ns(),
            start_native_tid=threading.get_native_id(),
            compact=True,
        )
        if parent_span_id is not None:
            fields["parent_span_id"] = parent_span_id
        context["span_id"] = span_id
        token = _CONTEXT.set(context)
    except Exception:
        start_ns = None
    status = "ok"
    try:
        # Callers may add completion metadata without changing the interval.
        yield fields
    except BaseException:
        status = "error"
        raise
    finally:
        try:
            if start_ns is not None:
                end_ns = time.monotonic_ns()
                fields.update(
                    end_ns=end_ns,
                    duration_ns=end_ns - start_ns,
                    status=status,
                )
                emit(event, **fields)
        except Exception:
            pass
        finally:
            if token is not None:
                with suppress(Exception):
                    _CONTEXT.reset(token)


def span(event: str, **fields: Any):
    return _span(event, fields)


@contextmanager
def measure_elapsed(fields: dict[str, Any] | None, name: str):
    """Record one host phase in caller-owned state without emitting a log."""
    if fields is None:
        yield
        return
    start = None
    with suppress(Exception):
        start = time.perf_counter()
    try:
        yield
    except BaseException:
        with suppress(Exception):
            fields["status"] = "error"
            if fields.get("failed_stage") is None:
                fields["failed_stage"] = name
        raise
    finally:
        with suppress(Exception):
            if start is not None:
                fields[name] = time.perf_counter() - start


def batch_fields(scheduler_output: Any) -> dict[str, Any]:
    """Snapshot existing IDs only; never retain a mutable scheduler payload."""
    return {
        "execution_id": scheduler_output.ttft_execution_id,
        "request_ids": tuple(scheduler_output.num_scheduled_tokens),
        "timing_scope": "shared_batch_host",
    }
