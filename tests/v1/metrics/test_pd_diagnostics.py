# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging

import pytest

from vllm.entrypoints.serve.utils.pd_diagnostics import PDRequestMiddleware
from vllm.v1.metrics import pd_diagnostics as diag


def test_exception_preserves_origin_and_restores_request_context(caplog):
    failure = ValueError("business failure")
    caplog.set_level(logging.INFO, logger=diag.__name__)
    with diag.scope(request_id="parent"):
        with pytest.raises(ValueError) as caught:
            with diag.scope(request_id="child"), diag.span("forward"):
                raise failure
        assert caught.value is failure
        assert diag.current() == {"request_id": "parent"}
    assert diag.current() == {}
    record = json.loads(caplog.records[-1].getMessage().split("[PD_DIAG] ", 1)[1])
    assert record["request_id"] == "child"
    assert record["status"] == "error"
    assert record["error_type"] == "ValueError"


def test_concurrent_requests_do_not_inherit_each_others_identity():
    async def observe(identity, entered, release):
        with diag.scope(request_id=identity):
            entered.set()
            await release.wait()
            return diag.current()["request_id"]

    async def run():
        first, second, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        a = asyncio.create_task(observe("A", first, release))
        b = asyncio.create_task(observe("B", second, release))
        await first.wait()
        await second.wait()
        release.set()
        assert await asyncio.gather(a, b) == ["A", "B"]
        assert diag.current() == {}

    asyncio.run(run())


def test_cancelled_span_keeps_cancellation_and_resets_context(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)

    async def run():
        failure = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError) as caught:
            with diag.scope(request_id="cancelled"), diag.span("engine_submit"):
                raise failure
        assert caught.value is failure
        assert diag.current() == {}

    asyncio.run(run())
    record = json.loads(caplog.records[-1].getMessage().split("[PD_DIAG] ", 1)[1])
    assert record["request_id"] == "cancelled"
    assert record["status"] == "cancelled"


@pytest.mark.parametrize(
    ("root_path", "path"),
    [
        ("", "/v1/chat/completions"),
        ("/prefix", "/prefix/v1/chat/completions"),
        ("/prefix", "/v1/chat/completions"),
    ],
)
def test_api_observations_survive_root_path_mount(caplog, root_path, path):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    response = [
        {"type": "http.response.start", "status": 200},
        {"type": "http.response.body", "body": b"{}"},
    ]
    received = []

    async def app(scope, receive, send):
        for message in response:
            await send(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        received.append(message)

    scope = {
        "type": "http",
        "path": path,
        "root_path": root_path,
        "method": "POST",
        "state": {},
    }
    asyncio.run(PDRequestMiddleware(app)(scope, receive, send))
    records = [
        json.loads(record.getMessage().split("[PD_DIAG] ", 1)[1])
        for record in caplog.records
        if "[PD_DIAG] " in record.getMessage()
    ]
    ingress = next(r for r in records if r["event"] == "api_ingress")
    terminal = next(r for r in records if r["event"] == "api_terminal")
    assert terminal["status"] == "completed"
    assert ingress["observation_id"] == terminal["observation_id"]
    assert received == response
