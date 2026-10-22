# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side renderer call boundaries, including intentionally skipped work."""

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from inspect import iscoroutinefunction
from typing import ParamSpec, TypeVar

from vllm.v1.metrics import ttft_diagnostics as diagnostics

_P = ParamSpec("_P")
_R = TypeVar("_R")


def trace_renderer(event: str, *, route: str | None = None):
    """Correlate a render call (which may fan out into several engine inputs)."""

    @contextmanager
    def trace(args):
        frontend_id = diagnostics.current().get("frontend_id")
        if event == "frontend_render" or frontend_id is None:
            frontend_id = diagnostics.new_id("frontend")
        fields = {"boundary": "frontend_host_call"}
        if route is not None:
            fields["route"] = route
        if event == "frontend_tokenize":
            prompt = args[1]
            if "prompt_token_ids" in prompt:
                fields.update(execution="skipped", reason="supplied_token_ids")
            elif "prompt_embeds" in prompt:
                fields.update(execution="skipped", reason="prompt_embeds")
            else:
                fields["execution"] = "executed"
        with diagnostics.scope(frontend_id=frontend_id), diagnostics.span(event, **fields):
            yield

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if not diagnostics.enabled():
                    return await fn(*args, **kwargs)
                with trace(args):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not diagnostics.enabled():
                return fn(*args, **kwargs)
            with trace(args):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def mark_engine_input(engine_input, *, route: str | None = None) -> None:
    """Attach correlation only; do not introduce or override arrival timestamps."""
    if not diagnostics.enabled():
        return
    frontend_id = diagnostics.current().get("frontend_id") or diagnostics.new_id(
        "frontend"
    )
    engine_input["ttft_frontend_id"] = frontend_id
    if route is not None:
        diagnostics.emit(
            "frontend_input",
            frontend_id=frontend_id,
            route=route,
            execution="skipped",
            reason="renderer_bypass",
            boundary="frontend_host_call",
        )
        diagnostics.emit(
            "frontend_tokenize",
            frontend_id=frontend_id,
            route=route,
            execution="skipped",
            reason="forwarded_token_ids" if route == "forwarded_tokens" else "harmony",
            boundary="frontend_host_call",
        )
