# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side diagnostic boundaries; never synchronize a device for timing."""

from contextlib import nullcontext
from copy import copy
from functools import wraps
from time import monotonic_ns

from vllm.v1.metrics import ttft_diagnostics as diag
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)

_DISABLED_SPAN = nullcontext()


def diagnostic_span(event):
    # Reuse the inert context when disabled: no context or field allocation.
    return diag.span(event) if diag.enabled() else _DISABLED_SPAN


def diagnostic_stage(event):
    """Measure an existing method, including its early returns and exceptions."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not diag.enabled():
                return fn(*args, **kwargs)
            with diag.span(event, boundary=fn.__qualname__):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def attach_step(output, context):
    """Keep empty/None outputs empty and leave the shared singleton untouched."""
    if isinstance(output, ModelRunnerOutput):
        if output is EMPTY_MODEL_RUNNER_OUTPUT:
            output = copy(output)
        output.ttft_step_id = context.get("step_id")
    return output


def worker_call(event):
    """Capture identity at execute, not from mutable runner state at sample/output.

    WorkerWrapperBase owns this diagnostic-only side channel. It is replaced on
    every execute and consumed on sample, including PP/connector-only returns.
    """
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not diag.enabled():
                return fn(self, *args, **kwargs)
            if event == "worker_execute":
                scheduler_output = args[0] if args else kwargs["scheduler_output"]
                runner = getattr(self.worker, "model_runner", None)
                runner_type = type(runner) if runner is not None else None
                context = {
                    "step_id": scheduler_output.ttft_step_id,
                    "runner": (f"{runner_type.__module__}.{runner_type.__qualname__}"
                               if runner_type is not None else None),
                    "rank": getattr(self.worker, "rank", None),
                    "request_ids": list(scheduler_output.num_scheduled_tokens),
                }
                self._ttft_pending_context = None
            else:
                context = self.__dict__.pop("_ttft_pending_context", None) or {
                    "step_id": None,
                    "runner": None,
                    "rank": getattr(self.worker, "rank", None),
                }
            self._ttft_last_context = context
            with diag.scope(**context), diag.span(event):
                output = fn(self, *args, **kwargs)
                if event == "worker_execute":
                    self._ttft_pending_context = context
                return attach_step(output, context)
        return wrapped
    return decorate


def output_context(worker):
    # Read __dict__ directly: WorkerWrapperBase.__getattr__ forwards to worker.
    return worker.__dict__.get("_ttft_last_context", {})


def get_output(output, context):
    """Resolve at the original caller's boundary, using its captured identity."""
    if not diag.enabled():
        return output.get_output()
    with diag.scope(**context), diag.span("worker_output"):
        return attach_step(output.get_output(), context)


def emit_queue_residence(context, start_ns):
    end_ns = monotonic_ns()
    with diag.scope(**context):
        diag.emit(
            "worker_output_queue", start_ns=start_ns, end_ns=end_ns,
            duration_ns=end_ns - start_ns, status="ok",
        )
