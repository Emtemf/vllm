# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side P/D observations; overlapping spans are not additive latency."""

import contextlib
import contextvars
import itertools
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

_LOG = logging.getLogger(__name__)
_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "pd_diagnostics", default=None
)
_RUN = uuid.uuid4().hex
_HOST = socket.gethostname()
_COUNTER = itertools.count()

_STAGE_PATHS = {
    "api_ingress": "request.api.ingress",
    "api_resolve": "request.api.resolve",
    "api_handler": "request.api.handler",
    "render": "request.api.render",
    "tokenize": "request.api.tokenize",
    "engine_submit": "request.engine.submit",
    "scheduler_schedule": "engine.schedule",
    "scheduler_external_lookup": "engine.schedule.external_lookup",
    "scheduler_queue_episode": "request.engine.queue",
    "request_hash_ranges": "request.block_hash",
    "engine_scheduler_update": "engine.update",
    "runner.forward": "worker.forward",
    "runner.sampling": "worker.sample",
    "runner.draft": "worker.draft",
    "kv.load_submit": "worker.kv.load_submit",
    "kv.save_wait": "worker.kv.save_wait",
    "kv.completion_observed": "worker.kv.completion",
    "pool_save_fence": "pool.save.fence",
    "request_summary": "request.summary",
    "frontend_request_terminal": "request.output",
}


def new_id(kind: str) -> str:
    return f"{kind}-{_RUN}-{os.getpid()}-{next(_COUNTER)}"


def current() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})


@contextlib.contextmanager
def scope(**fields: Any) -> Iterator[None]:
    values = current()
    values.update(fields)
    token = _CONTEXT.set(values)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def emit(event: str, **fields: Any) -> None:
    """Emit a completed observation without replacing a business failure."""
    try:
        values = current()
        values.update(fields)
        if event == "pool_task":
            stage_path = "pool." + str(values.get("operation", "task"))
        elif event == "scheduler_kv_range":
            stage_path = "engine.schedule." + str(values.get("stage", "kv"))
        elif event.startswith("pd_transfer_"):
            stage_path = "pd.transfer." + event.removeprefix("pd_transfer_")
        else:
            stage_path = _STAGE_PATHS.get(event, event.replace("_", "."))
        values.update(
            event=event,
            version=1,
            host=_HOST,
            pid=os.getpid(),
            native_tid=threading.get_native_id(),
            run_id=_RUN,
            timestamp_ns=time.monotonic_ns(),
            record_id=new_id("record"),
            stage_path=stage_path,
        )
        _LOG.info("[PD_DIAG] %s", json.dumps(values, separators=(",", ":")))
    except Exception:
        # Diagnostic serialization and logging must not affect inference.
        pass


@contextlib.contextmanager
def span(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Measure the original host call; never synchronize the device."""
    values = dict(fields)
    start = time.monotonic_ns()
    values["start_ns"] = start
    values["status"] = "ok"
    try:
        yield values
    except BaseException as exc:
        values["status"] = (
            "cancelled"
            if isinstance(exc, (KeyboardInterrupt, GeneratorExit))
            or type(exc).__name__ == "CancelledError"
            else "error"
        )
        values["error_type"] = type(exc).__name__
        raise
    finally:
        end = time.monotonic_ns()
        values["end_ns"] = end
        values["duration_ns"] = end - start
        emit(event, **values)
