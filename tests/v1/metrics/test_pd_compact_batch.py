# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise host-only diagnostic ownership without loading device runtimes."""

import ast
import asyncio
import importlib.util
import json
import logging
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "compact_batch_subject", ROOT / "vllm/v1/metrics/pd_diagnostics.py"
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def records(caplog):
    return [
        json.loads(r.getMessage().split("[PD_DIAG] ", 1)[1])
        for r in caplog.records
        if "[PD_DIAG] " in r.getMessage()
    ]


def test_repeated_stage_sums_calls_without_counting_idle_envelope(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    with diag.batch("worker_summary", execution_id="step-1") as owner:
        owner.add_stage("forward", start_ns=10, end_ns=20)
        owner.add_stage("forward", start_ns=100, end_ns=130)
        owner.add_stage("sampling", duration_ns=0)
    row = records(caplog)[0]
    assert row["stages"]["forward"]["duration_ns"] == 40
    assert row["stages"]["forward"]["calls"] == 2
    assert row["stages"]["forward"]["end_ns"] == 130
    assert row["stages"]["sampling"]["duration_ns"] == 0


@pytest.mark.parametrize("skip_first", [False, True])
def test_skipped_call_does_not_erase_real_measurements(caplog, skip_first):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    owner = diag.Batch("worker_summary")
    calls = [{"duration_ns": 7}, {"status": "skipped"}]
    for values in reversed(calls) if skip_first else calls:
        owner.add_stage("sampling", **values)
    owner.finish()
    stage = records(caplog)[0]["stages"]["sampling"]
    assert (stage["duration_ns"], stage["status"]) == (7, "mixed")
    assert (stage["measured_calls"], stage["skipped_calls"]) == (1, 1)


def test_missing_call_cannot_be_reported_as_complete_measured_total(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    with diag.batch("worker_summary") as owner:
        owner.add_stage("forward", duration_ns=8)
        owner.add_stage("forward", status="missing")
        owner.add_stage("forward", duration_ns=9)
    stage = records(caplog)[0]["stages"]["forward"]
    assert stage["duration_ns"] is None
    assert stage["status"] == "missing"
    assert stage["unmeasured_calls"] == 1


def test_later_success_does_not_erase_earlier_call_failure(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    with diag.batch("worker_summary") as owner:
        owner.add_stage("forward", duration_ns=8, status="error")
        owner.add_stage("forward", duration_ns=9)
        owner.add_stage("forward", status="skipped")
    stage = records(caplog)[0]["stages"]["forward"]
    assert (stage["duration_ns"], stage["status"]) == (17, "error")
    assert stage["exceptions"] == 1


@pytest.mark.parametrize("failure", [ValueError("business"), asyncio.CancelledError()])
def test_original_failure_and_context_survive_summary(caplog, failure):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    with diag.scope(request_id="outer"):
        with (
            pytest.raises(type(failure)) as caught,
            diag.scope(request_id="inner"),
            diag.batch("worker_summary"),
            diag.span("forward"),
        ):
            raise failure
        assert caught.value is failure
        assert diag.current() == {"request_id": "outer"}
        assert diag.current_batch() is None
    row = records(caplog)[0]
    assert row["request_id"] == "inner"
    assert row["status"] == (
        "cancelled" if isinstance(failure, asyncio.CancelledError) else "error"
    )
    assert row["stages"]["forward"]["exceptions"] == 1


def test_shared_output_handler_restores_explicit_owner(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)

    async def run():
        a = diag.Batch("api_summary", request_id="A")
        b = diag.Batch("api_summary", request_id="B")

        async def output():
            await asyncio.sleep(0)
            with diag.collect(b):
                diag.emit("request_summary", ttft=2.5)
            diag.emit("request_summary", ttft=1.5)

        with diag.collect(a):
            await asyncio.create_task(output())
        a.finish()
        b.finish()

    asyncio.run(run())
    assert {r["request_id"]: r["official"]["ttft"] for r in records(caplog)} == {
        "A": 1.5,
        "B": 2.5,
    }


def test_bounded_links_and_finish_are_explicit_and_idempotent(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    owner = diag.Batch("core_summary")
    for number in range(130):
        owner.add_link("member", request_id=str(number))
    owner.finish()
    owner.add_stage("schedule", duration_ns=99)
    owner.finish()
    rows = records(caplog)
    assert len(rows) == 1
    assert rows[0]["complete"] is False
    assert rows[0]["dropped_observations"] == 2
    assert rows[0]["links"][-1]["request_id"] == "127"
    assert rows[0]["stages"] == {}


def test_logging_failure_never_replaces_business_failure(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("logger unavailable")

    monkeypatch.setattr(diag._LOG, "info", broken)
    failure = RuntimeError("original")
    with (
        pytest.raises(RuntimeError) as caught,
        diag.batch("worker_summary"),
        diag.span("forward"),
    ):
        raise failure
    assert caught.value is failure


def worker_decorators():
    # Compile the real host wrappers in isolation from torch/CUDA imports.
    tree = ast.parse((ROOT / "vllm/v1/worker/worker_base.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("pd_execution", "pd_sample")
    ]
    namespace = {
        "diag": diag,
        "wraps": wraps,
        "Any": Any,
        "pd_execution_context": lambda owner, output: {
            "execution_id": output.execution_id,
            "request_ids": list(output.requests),
            "tp_rank": 0,
        },
    }
    exec(compile(tree, "worker_base.py", "exec"), namespace)
    return namespace["pd_execution"], namespace["pd_sample"]


def test_execute_sample_pair_retains_origin_without_ambient_request(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    execute, sample = worker_decorators()

    class Runner:
        @execute
        def execute_model(self, output):
            with diag.span("forward"):
                return "deferred"

        @sample
        def sample_tokens(self):
            with diag.span("sampling"):
                return "tokens"

    runner = Runner()
    assert (
        runner.execute_model(SimpleNamespace(execution_id="E1", requests=["R1"]))
        == "deferred"
    )
    with diag.scope(execution_id="unrelated"):
        assert runner.sample_tokens() == "tokens"
    rows = records(caplog)
    assert [(r["phase"], r["execution_id"], r["request_ids"]) for r in rows] == [
        ("execute", "E1", ["R1"]),
        ("sample", "E1", ["R1"]),
    ]
    assert runner._pd_execution_context is None
