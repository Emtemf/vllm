# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, process-local TTFT summaries; never synchronize business work."""

import contextlib
import contextvars
import itertools
import json
import logging
import math
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
_BATCH: contextvars.ContextVar["Batch | None"] = contextvars.ContextVar(
    "pd_compact_batch", default=None
)
_RUN = uuid.uuid4().hex
_HOST = socket.gethostname()
_COUNTER = itertools.count()
_MAX_ITEMS = 128
_SUMMARIES = {
    "api_summary",
    "core_summary",
    "worker_summary",
    "pool_task",
    "pd_transfer_summary",
}
_SPANS = {
    "api_resolve": "req_resolve",
    "req_resolve": "req_resolve",
    "render": "render",
    "tokenize": "tokenize",
    "add_request": "add_request",
    "scheduler_schedule": "schedule",
    "schedule": "schedule",
    "runner.forward": "forward",
    "forward": "forward",
    "runner.sampling": "sampling",
    "sampling": "sampling",
    "kv.load_submit": "submit",
    "submit": "submit",
    "kv.save_wait": "wait_for_save",
    "wait_for_save": "wait_for_save",
    "wait_for_transfer": "wait_for_transfer",
}
_INTERNAL_SPANS = {"scheduler_external_lookup"}
_LINK_FIELDS = {
    "observation_id",
    "api_request_id",
    "external_req_id",
    "request_id",
    "request_ids",
    "parent_request_id",
    "choice_index",
    "prompt_index",
    "execution_id",
    "task_id",
    "store_job_id",
    "engine_id",
    "remote_engine_id",
    "remote_request_id",
    "transfer_request_id",
    "transfer_engine_id",
    "transfer_role",
    "kv_role",
    "dp_rank",
    "pp_rank",
    "tp_rank",
    "finished_sending",
    "finished_recving",
    "failed_recving",
}
_LINK_EVENTS = {
    "api_request_link",
    "request_id_link",
    "request_child_link",
    "frontend_request_registered",
    "pd_request_link",
    "pd_transfer_offer",
    "kv.completion_observed",
    "engine_execution",
    "scheduler_execution",
}


def new_id(kind: str) -> str:
    return f"{kind}-{_RUN}-{os.getpid()}-{next(_COUNTER)}"


def clock_domain() -> dict[str, Any]:
    """Identity of the process-local monotonic clock used by frontend carriers."""
    return {"host": _HOST, "pid": os.getpid(), "run_id": _RUN}


def current() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})


def current_batch() -> "Batch | None":
    value = _BATCH.get()
    return value if value is not None and not value.closed else None


@contextlib.contextmanager
def scope(**fields: Any) -> Iterator[None]:
    values = current()
    values.update(fields)
    token = _CONTEXT.set(values)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@contextlib.contextmanager
def collect(value: "Batch | None") -> Iterator["Batch | None"]:
    """Temporarily restore an explicit owner, including a shared output task."""
    token = _BATCH.set(value)
    try:
        yield value
    finally:
        _BATCH.reset(token)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _write(event: str, fields: dict[str, Any]) -> None:
    try:
        values = dict(fields)
        values.update(
            event=event,
            version=2,
            host=_HOST,
            pid=os.getpid(),
            native_tid=threading.get_native_id(),
            run_id=_RUN,
            timestamp_ns=time.monotonic_ns(),
            record_id=new_id("record"),
        )
        _LOG.info("[PD_DIAG] %s", json.dumps(values, separators=(",", ":")))
    except Exception:
        # Diagnostics must not replace a serving failure or fail a response.
        pass


class Batch:
    """One original lifecycle's bounded timing/identity accumulator."""

    def __init__(self, event: str, **identity: Any):
        if event not in _SUMMARIES:
            raise ValueError(f"Not a compact summary type: {event}")
        self.event = event
        self.fields = {**current(), **identity}
        self.stages: dict[str, dict[str, Any]] = {}
        self.links: list[dict[str, Any]] = []
        self.closed = False
        self._lock = threading.RLock()
        self._dropped = 0

    def update(self, **fields: Any) -> None:
        with self._lock:
            if not self.closed:
                self.fields.update(fields)

    def add_stage(
        self,
        name: str,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
        duration_ns: int | None = None,
        calls: int = 1,
        status: str = "ok",
        **attributes: Any,
    ) -> None:
        if duration_ns is None and _number(start_ns) and _number(end_ns):
            duration_ns = end_ns - start_ns
        exceptions = attributes.pop("exceptions", None)
        failed = status in ("error", "failed", "cancelled", "exception")
        with self._lock:
            if self.closed:
                return
            if name not in self.stages and len(self.stages) >= _MAX_ITEMS:
                self._dropped += 1
                return
            value = self.stages.setdefault(
                name,
                {
                    "calls": 0,
                    "duration_ns": 0,
                    "start_ns": None,
                    "end_ns": None,
                    "status": status,
                    "exceptions": 0,
                    "measured_calls": 0,
                    "skipped_calls": 0,
                    "unmeasured_calls": 0,
                },
            )
            count = calls if _number(calls) else 1
            value["calls"] += count
            if status in ("skipped", "fused"):
                value["skipped_calls"] += count
                if not value["measured_calls"]:
                    value["status"] = status
                    value["duration_ns"] = None
                elif not value["unmeasured_calls"] and not value["exceptions"]:
                    value["status"] = "mixed"
            elif status in ("missing", "invalid") or not _number(duration_ns):
                value["unmeasured_calls"] += max(count, 1)
                value["status"] = (
                    status
                    if status in ("missing", "invalid")
                    else "invalid"
                    if duration_ns is not None
                    else "missing"
                )
                value["duration_ns"] = None
            else:
                value["measured_calls"] += count
                if not value["unmeasured_calls"]:
                    value["duration_ns"] = (value["duration_ns"] or 0) + duration_ns
                    if failed or not value["exceptions"]:
                        value["status"] = (
                            "mixed"
                            if status == "ok" and value["skipped_calls"]
                            else status
                        )
            if _number(start_ns):
                old = value["start_ns"]
                value["start_ns"] = start_ns if old is None else min(old, start_ns)
            if _number(end_ns):
                old = value["end_ns"]
                value["end_ns"] = end_ns if old is None else max(old, end_ns)
            value["exceptions"] += (
                max(exceptions, int(failed)) if _number(exceptions) else int(failed)
            )
            value.update(attributes)

    def add_link(self, kind: str, **identity: Any) -> None:
        link = {"kind": kind, **identity}
        with self._lock:
            if self.closed or link in self.links:
                return
            if len(self.links) >= _MAX_ITEMS:
                self._dropped += 1
            else:
                self.links.append(link)

    def capture(self, event: str, **fields: Any) -> None:
        """Merge retained observations, not old leaf-record stdout."""
        stage = _SPANS.get(event)
        if stage is not None:
            values = dict(fields)
            values.pop("event", None)
            self.add_stage(stage, **values)
        elif event == "request_summary":
            self.update(official=dict(fields))
        elif event == "frontend_request_terminal":
            duration = fields.get("detokenize_duration_ns")
            if duration is not None:
                calls = fields.get("detokenize_calls", 1)
                self.add_stage(
                    "detokenize",
                    duration_ns=duration,
                    calls=calls,
                    status=(
                        "skipped"
                        if calls == 0
                        else "ok"
                        if fields.get("error_type") is None
                        else "error"
                    ),
                )
            self.update(terminal=dict(fields))
        elif event in _LINK_EVENTS:
            self.add_link(
                event, **{k: v for k, v in fields.items() if k in _LINK_FIELDS}
            )

    def finish(self, status: str = "ok", **fields: Any) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            values = {
                **self.fields,
                **fields,
                "stages": self.stages,
                "links": self.links,
                "status": status,
                "complete": (
                    fields.get("complete", self.fields.get("complete", True))
                    and not self._dropped
                ),
                "dropped_observations": self._dropped,
            }
        _write(self.event, values)


@contextlib.contextmanager
def batch(event: str, **identity: Any) -> Iterator[Batch]:
    value = Batch(event, **identity)
    status = "ok"
    error_type = None
    try:
        with collect(value):
            yield value
    except BaseException as exc:
        status = (
            "cancelled"
            if isinstance(exc, (KeyboardInterrupt, GeneratorExit))
            or type(exc).__name__ == "CancelledError"
            else "error"
        )
        error_type = type(exc).__name__
        raise
    finally:
        value.finish(status=status, error_type=error_type)


def emit(event: str, **fields: Any) -> None:
    if event in _SUMMARIES:
        _write(event, {**current(), **fields})
        return
    value = current_batch()
    if value is not None:
        value.capture(event, **fields)


@contextlib.contextmanager
def span(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Measure retained host calls and necessary scheduler formula inputs."""
    if event not in _SPANS and event not in _INTERNAL_SPANS:
        yield {}
        return
    values = dict(fields)
    start = time.monotonic_ns()
    values.update(start_ns=start, status="ok")
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
        values.update(end_ns=end, duration_ns=end - start)
        value = current_batch()
        if value is not None and event in _SPANS:
            value.capture(event, **values)
