# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive request boundaries; never consume bodies or inspect streamed bytes."""

import asyncio
from contextvars import ContextVar
from time import monotonic_ns
from typing import Any

from vllm.v1.metrics import pd_diagnostics as diag

_observation: ContextVar[dict[str, Any] | None] = ContextVar(
    "pd_api_observation", default=None
)


def install(app) -> None:
    if not getattr(app.state, "pd_diagnostics_installed", False):
        app.add_middleware(PDRequestMiddleware)
        app.state.pd_diagnostics_installed = True


def typed_handler(request, *, stream: bool) -> None:
    state = _observation.get()
    if state is None or state["typed"]:
        return
    state["typed"] = True
    state["stream"] = stream
    now = monotonic_ns()
    state["batch"].add_stage(
        "req_resolve",
        start_ns=state["start_ns"],
        end_ns=now,
        source="asgi_ingress_to_typed_handler",
    )
    state["batch"].update(stream=stream)


def request_link(api_request_id: str, external_req_id: str, index: int) -> None:
    owner = diag.current_batch()
    if owner is not None:
        owner.add_link(
            "api_request",
            api_request_id=api_request_id,
            external_req_id=external_req_id,
            prompt_index=index,
        )


def generated(
    request_id: str, *, kind: str, stream: bool, choice_index: int | None = None
) -> None:
    state = _observation.get()
    if state is None or state.get("generated"):
        return
    state["generated"] = True
    state["batch"].update(
        first_generated={
            "api_request_id": request_id,
            "output_kind": kind,
            "stream": stream,
            "choice_index": choice_index,
            "timestamp_ns": monotonic_ns(),
            "source": "serving_generated_event" if stream else "full_response_assembly",
        }
    )


def generated_chat(
    request_id: str, message, *, stream: bool, choice_index: int | None = None
) -> None:
    state = _observation.get()
    if state is None or state.get("generated") or message is None:
        return
    for field in ("content", "reasoning", "tool_calls"):
        if getattr(message, field, None):
            generated(request_id, kind=field, stream=stream, choice_index=choice_index)
            return


def generation_error(error: BaseException) -> None:
    state = _observation.get()
    if state is not None:
        state["generation_error"] = type(error).__name__
        if isinstance(error, asyncio.CancelledError):
            state["cancelled"] = True


class PDRequestMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if scope["type"] != "http" or path not in (
            "/v1/chat/completions",
            "/v1/completions",
        ):
            return await self.app(scope, receive, send)
        observation_id = diag.new_id("api")
        start = monotonic_ns()
        owner = diag.Batch(
            "api_summary",
            scope="http",
            observation_id=observation_id,
            route=scope["path"],
            method=scope.get("method"),
            start_ns=start,
        )
        state: dict[str, Any] = {"start_ns": start, "typed": False, "batch": owner}
        scope.setdefault("state", {})["pd_observation_id"] = observation_id
        token = _observation.set(state)
        status_code = None
        terminal = False
        disconnected = False
        error_type = None
        cancelled = False

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal status_code, terminal
            await send(message)
            if message["type"] == "http.response.start":
                status_code = message["status"]
                owner.update(
                    response_start={
                        "http_status": status_code,
                        "timestamp_ns": monotonic_ns(),
                    }
                )
            elif message["type"] == "http.response.body":
                if message.get("body") and not state.get("first_body"):
                    state["first_body"] = True
                    owner.update(
                        first_body={
                            "stream": state.get("stream"),
                            "timestamp_ns": monotonic_ns(),
                            "source": "asgi_send_return_not_client_receipt",
                        }
                    )
                if not message.get("more_body", False):
                    terminal = True

        try:
            with diag.scope(observation_id=observation_id), diag.collect(owner):
                try:
                    return await self.app(scope, observed_receive, observed_send)
                except asyncio.CancelledError as exc:
                    cancelled = True
                    error_type = type(exc).__name__
                    raise
                except BaseException as exc:
                    error_type = type(exc).__name__
                    raise
                finally:
                    end = monotonic_ns()
                    status = (
                        "cancelled"
                        if cancelled or state.get("cancelled")
                        else "error"
                        if error_type or state.get("generation_error")
                        else "rejected"
                        if status_code is not None and status_code >= 400
                        else "completed"
                        if terminal
                        else "disconnected"
                        if disconnected
                        else "incomplete"
                    )
                    if not state["typed"]:
                        owner.add_stage(
                            "req_resolve",
                            start_ns=start,
                            end_ns=end,
                            status=status,
                            source="asgi_ingress_to_framework_terminal",
                            typed_handler_entered=False,
                        )
                    owner.finish(
                        status=status,
                        end_ns=end,
                        http_status=status_code,
                        complete=terminal and status in ("completed", "rejected"),
                        completion="http_terminal",
                        response_complete=terminal,
                        disconnect_observed=disconnected,
                        error_type=error_type or state.get("generation_error"),
                    )
        finally:
            _observation.reset(token)
