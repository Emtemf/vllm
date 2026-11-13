# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema-2 producer behavior without importing the vLLM or torch runtime.

Execute the production definitions with small boundary objects. This keeps the
ASGI lifecycle, request ownership, original stats formulas and output order real.
The compact helper's own aggregation tests live separately.
"""

import ast
import asyncio
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from threading import get_ident, get_native_id
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def definitions(monkeypatch, name, path, names, **bindings):
    """Load real selected definitions, excluding runtime dependency imports."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in names:
                continue
            methods = names[node.name]
            if isinstance(node, ast.ClassDef) and methods is not None:
                node.body = [
                    member
                    for member in node.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and member.name in methods
                    and not any(
                        isinstance(decorator, ast.Name) and decorator.id == "overload"
                        for decorator in member.decorator_list
                    )
                ]
                node.bases = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    monkeypatch.setitem(sys.modules, name, module)
    module.__dict__.update(bindings)
    code = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(code, str(ROOT / path), "exec"), module.__dict__)
    return module


class FinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3

    def __str__(self):
        return ("stop", "length", "abort", "error")[self.value]


@pytest.fixture
def api(monkeypatch, caplog):
    spec = importlib.util.spec_from_file_location(
        "compact_api_test_diag", ROOT / "vllm/v1/metrics/pd_diagnostics.py"
    )
    diag = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, diag)
    spec.loader.exec_module(diag)
    caplog.set_level(logging.INFO, logger=diag.__name__)
    http_spec = importlib.util.spec_from_file_location(
        "compact_api_test_http", ROOT / "vllm/entrypoints/serve/utils/pd_diagnostics.py"
    )
    http = importlib.util.module_from_spec(http_spec)
    # Only this module import is replaced; no vLLM package is initialized.
    tree = ast.parse(Path(http_spec.origin).read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom) or node.module != "vllm.v1.metrics"
    ]
    http.diag = diag
    exec(compile(tree, http_spec.origin, "exec"), http.__dict__)
    stats = definitions(
        monkeypatch,
        "compact_api_test_stats",
        "vllm/v1/metrics/stats.py",
        {
            "RequestStateStats": None,
            "FinishedRequestStats": None,
            "PromptTokenStats": None,
            "IterationStats": None,
        },
        dataclass=dataclass,
        field=field,
        time=time,
        diag=diag,
        envs=SimpleNamespace(VLLM_COMPUTE_NANS_IN_LOGITS=False),
    )
    output = definitions(
        monkeypatch,
        "compact_api_test_output",
        "vllm/v1/engine/output_processor.py",
        {
            "RequestState": {"__init__"},
            "OutputProcessorOutput": None,
            "OutputProcessor": {
                "__init__",
                "process_outputs",
                "_emit_request_terminal",
                "_finish_pd_request",
                "_finish_request",
                "_update_admission_stats",
                "get_num_unfinished_requests",
                "_trim_surplus_tokens",
                "_update_stats_from_output",
                "_update_stats_from_finished",
                "abort_requests",
                "propagate_error",
            },
        },
        diag=diag,
        asyncio=asyncio,
        monotonic_ns=time.monotonic_ns,
        get_ident=get_ident,
        get_native_id=get_native_id,
        dataclass=dataclass,
        deque=deque,
        defaultdict=defaultdict,
        FinishReason=FinishReason,
        RequestStateStats=stats.RequestStateStats,
        length_from_prompt_token_ids_or_embeds=lambda tokens, embeds: len(tokens or []),
        LoRARequestStates=lambda enabled: SimpleNamespace(
            request_finished=lambda *args: None
        ),
        ParentRequest=SimpleNamespace(observe_finished_request=lambda *args: None),
        EMPTY_CPU_TENSOR=object(),
    )
    engine = definitions(
        monkeypatch,
        "compact_api_test_engine",
        "vllm/v1/engine/async_llm.py",
        {
            "_api_submission": None,
            "AsyncLLM": {
                "_record_pd_failure",
                "_send_engine_request",
                "_add_request",
                "encode",
            },
        },
        diag=diag,
        asyncio=asyncio,
        time=SimpleNamespace(monotonic_ns=lambda: diag.time.monotonic_ns()),
        contextmanager=contextmanager,
        VLLMClientError=type("VLLMClientError", (Exception,), {}),
        GracefulHTTPError=type("GracefulHTTPError", (Exception,), {}),
        EngineDeadError=type("EngineDeadError", (Exception,), {}),
        EngineGenerateError=type("EngineGenerateError", (Exception,), {}),
    )
    return SimpleNamespace(
        diag=diag, http=http, stats=stats, output=output, engine=engine
    )


def records(caplog):
    return [
        json.loads(record.getMessage().split("[PD_DIAG] ", 1)[1])
        for record in caplog.records
        if "[PD_DIAG] " in record.getMessage()
    ]


def state(
    api,
    request_id="internal",
    *,
    parent=None,
    index=0,
    log_stats=True,
    pd_identity=None,
):
    return api.output.RequestState(
        request_id=request_id,
        external_req_id="external",
        parent_req=parent,
        request_index=index,
        lora_request=None,
        output_kind=None,
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=8,
        arrival_time=100.0,
        queue=None,
        log_stats=log_stats,
        stream_interval=1,
        pd_identity=pd_identity,
    )


def processor(api, states):
    result = api.output.OutputProcessor(None, log_stats=True)
    for value in states:
        result.request_states[value.request_id] = value
        result.external_req_ids[value.external_req_id].append(value.request_id)
        value.make_request_output = lambda *args, **kwargs: None
    return result


def output(request_id, tokens, finish=None, *, execution_id=None, output_batch_id=None):
    return SimpleNamespace(
        request_id=request_id,
        new_token_ids=tokens,
        pooling_output=None,
        finish_reason=finish,
        stop_reason=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        routed_experts=None,
        prefill_stats=None,
        spec_decode_metrics=None,
        mm_cache_miss_hashes=None,
        new_sampling_mask=None,
        finished=finish is not None,
        events=None,
        num_nans_in_logits=0,
        pd_execution_id=execution_id,
        output_batch_id=output_batch_id,
    )


@pytest.mark.parametrize(
    "root_path,path",
    [
        ("", "/v1/chat/completions"),
        ("/prefix", "/prefix/v1/chat/completions"),
        ("/prefix", "/v1/chat/completions"),
    ],
)
def test_http_and_submission_close_independently(api, caplog, root_path, path):
    sent = []
    saved = []

    async def app(scope, receive, send):
        api.http.typed_handler(object(), stream=True)
        api.http.request_link("chat-1", "external", 0)
        with api.engine._api_submission("external") as submission:
            saved.append(state(api))
            submission.add_link("engine_request", request_id="internal")
        assert submission.closed
        assert not saved[0].pd_batch.closed
        api.http.generated("chat-1", kind="content", stream=True)
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"data", "more_body": True})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        raise AssertionError("middleware must not consume the body")

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": path, "root_path": root_path, "method": "POST"}
    asyncio.run(api.http.PDRequestMiddleware(app)(scope, receive, send))
    emitted = records(caplog)
    assert [item["scope"] for item in emitted] == ["engine_submission", "http"]
    submission, http = emitted
    assert http["version"] == 2 and http["status"] == "completed"
    assert http["observation_id"] == submission["observation_id"]
    assert http["links"][-1]["submission_id"] == submission["submission_id"]
    assert saved[0].pd_batch.fields["submission_id"] == submission["submission_id"]
    assert http["first_generated"]["output_kind"] == "content"
    assert http["first_body"]["source"] == "asgi_send_return_not_client_receipt"
    assert http["stages"]["req_resolve"]["calls"] == 1
    assert "add_request" not in submission["stages"]
    assert "add_request" not in http["stages"]
    assert len(sent) == 3
    assert api.diag.current_batch() is None and api.diag.current() == {}
    api.output.OutputProcessor._emit_request_terminal(saved[0], "stop")
    assert records(caplog)[-1]["scope"] == "request"


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ValueError("original"), "error"),
        (asyncio.CancelledError("original"), "cancelled"),
    ],
)
def test_http_preserves_original_exception_and_context(api, caplog, failure, expected):
    async def app(scope, receive, send):
        raise failure

    async def unused(*args):
        raise AssertionError("no I/O expected")

    with api.diag.scope(request_id="outer"):
        with pytest.raises(type(failure)) as caught:
            asyncio.run(
                api.http.PDRequestMiddleware(app)(
                    {"type": "http", "path": "/v1/completions"}, unused, unused
                )
            )
        assert caught.value is failure
        assert api.diag.current() == {"request_id": "outer"}
    record = records(caplog)[-1]
    assert record["status"] == expected and not record["complete"]
    assert record["stages"]["req_resolve"]["typed_handler_entered"] is False
    assert record["error_type"] == type(failure).__name__
    assert api.http._observation.get() is None


def test_framework_rejection_and_caught_generation_error(api, caplog):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 422})
        await send({"type": "http.response.body", "body": b"{}"})

    async def unused(*args):
        return None

    asyncio.run(
        api.http.PDRequestMiddleware(app)(
            {"type": "http", "path": "/v1/completions"}, unused, unused
        )
    )
    assert records(caplog)[-1]["status"] == "rejected"
    assert records(caplog)[-1]["complete"] is True

    async def handled(scope, receive, send):
        api.http.typed_handler(object(), stream=True)
        api.http.generation_error(RuntimeError("caught by serving"))
        await app(scope, receive, send)

    asyncio.run(
        api.http.PDRequestMiddleware(handled)(
            {"type": "http", "path": "/v1/completions"}, unused, unused
        )
    )
    assert records(caplog)[-1]["status"] == "error"
    assert records(caplog)[-1]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "kind,expected",
    [("error", "error"), ("cancel", "cancelled"), ("reject", "rejected")],
)
def test_submission_error_is_original_and_http_owner_restored(
    api, caplog, kind, expected
):
    failure = {
        "error": ValueError("bad"),
        "cancel": asyncio.CancelledError(),
        "reject": api.engine.VLLMClientError("invalid"),
    }[kind]
    outer = api.diag.Batch("api_summary", scope="http", observation_id="http")
    with api.diag.collect(outer):
        with (
            pytest.raises(type(failure)) as caught,
            api.engine._api_submission("external"),
        ):
            raise failure
        assert caught.value is failure and api.diag.current_batch() is outer
    record = records(caplog)[-1]
    assert record["scope"] == "engine_submission" and record["status"] == expected
    assert record["complete"] is False and not outer.closed


def test_shared_output_context_never_owns_other_request(api, caplog):
    with api.diag.scope(observation_id="http-A", submission_id="submit-A"):
        first = state(api, "A")
    with api.diag.scope(observation_id="http-B", submission_id="submit-B"):
        second = state(api, "B")
    unrelated = api.diag.Batch("api_summary", scope="http", observation_id="wrong")
    with api.diag.collect(unrelated), api.diag.scope(request_id="wrong"):
        api.output.OutputProcessor._emit_request_terminal(first, "abort")
        api.output.OutputProcessor._emit_request_terminal(second, "error", "ValueError")
        api.output.OutputProcessor._emit_request_terminal(first, "error", "Duplicate")
    emitted = records(caplog)
    assert [
        (item["request_id"], item["observation_id"], item["submission_id"])
        for item in emitted
    ] == [("A", "http-A", "submit-A"), ("B", "http-B", "submit-B")]
    assert emitted[0]["stages"]["detokenize"]["status"] == "skipped"
    assert emitted[0]["stages"]["detokenize"]["duration_ns"] is None
    assert not unrelated.closed and not unrelated.stages


def test_parent_cancel_closes_each_child_before_original_abort(api, caplog):
    parent = SimpleNamespace(request_id="parent", child_requests={"child-0", "child-1"})
    children = [
        state(api, f"child-{index}", parent=parent, index=index) for index in range(2)
    ]
    proc = processor(api, children)
    proc.parent_requests["parent"] = parent
    engine = api.engine.AsyncLLM()
    engine.output_processor = proc
    failure = asyncio.CancelledError()
    engine._record_pd_failure("parent", failure)
    assert set(proc.abort_requests(["parent"], internal=True)) == {"child-0", "child-1"}
    emitted = records(caplog)
    assert len(emitted) == 2
    assert {item["choice_index"] for item in emitted} == {0, 1}
    assert all(item["parent_request_id"] == "parent" for item in emitted)
    assert all(item["status"] == "cancelled" for item in emitted)
    assert all(item["error_type"] == "CancelledError" for item in emitted)
    assert not proc.request_states and not proc.parent_requests


@pytest.mark.parametrize("cancelled", [False, True])
def test_pooling_failure_records_before_abort_and_preserves_exception(
    api, caplog, cancelled
):
    request = state(api)
    proc = processor(api, [request])
    engine = api.engine.AsyncLLM()
    engine.output_processor = proc
    engine.log_requests = False
    failure = asyncio.CancelledError() if cancelled else ValueError("pooling failed")
    expected = "cancelled" if cancelled else "error"
    events = []

    def fail():
        raise failure

    queue = SimpleNamespace(
        request_id="internal",
        get_nowait=fail,
        close=lambda: events.append("close"),
    )

    async def add_request(*args, **kwargs):
        return queue

    async def abort(request_id, *, internal):
        events.append("abort")
        emitted = records(caplog)
        assert len(emitted) == 1
        assert emitted[0]["status"] == expected
        assert emitted[0]["error_type"] == type(failure).__name__
        assert request.pd_batch.closed
        assert request_id == "internal" and internal is True
        proc.abort_requests([request_id], internal=internal)

    engine.add_request = add_request
    engine.abort = abort

    async def consume():
        async for _ in engine.encode("prompt", None, "external"):
            pytest.fail("failed pooling must not yield output")

    exception_type = (
        asyncio.CancelledError if cancelled else api.engine.EngineGenerateError
    )
    with pytest.raises(exception_type) as caught:
        asyncio.run(consume())
    if cancelled:
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure
    assert events == ["abort", "close"]
    assert not proc.request_states
    assert len(records(caplog)) == 1


def test_detokenize_sums_calls_and_preserves_official_endpoint(
    api, caplog, monkeypatch
):
    request = state(api)
    token_count = [0]

    def update(tokens, stopped):
        token_count[0] += len(tokens)
        return None

    request.detokenizer = SimpleNamespace(
        update=update, num_output_tokens=lambda: token_count[0]
    )
    request.logprobs_processor = SimpleNamespace(update_from_output=lambda value: None)
    proc = processor(api, [request])
    ticks = iter([10, 10, 100, 107, 1000, 1005])
    monkeypatch.setattr(api.output, "monotonic_ns", lambda: next(ticks))
    # First output observation has its own timestamp; keep timing deterministic.
    request.pd_first_output = True
    stats = api.stats.IterationStats()
    stats.iteration_timestamp = 101.0
    request.stats.scheduled_ts = 10.0
    request.stats.queued_ts = 9.0
    with api.diag.scope(execution_id="ambient-wrong"):
        proc.process_outputs(
            [output("internal", [3], execution_id="exec-1", output_batch_id="batch-1")],
            12.0,
            stats,
        )
    stats.iteration_timestamp = 104.0
    proc.process_outputs(
        [output("internal", [4], execution_id="exec-2", output_batch_id="batch-2")],
        14.0,
        stats,
    )
    stats.iteration_timestamp = 105.0
    proc.process_outputs([output("internal", [5], FinishReason.STOP)], 15.0, stats)
    record = records(caplog)[-1]
    timing = record["stages"]["detokenize"]
    assert timing["duration_ns"] == 12 and timing["calls"] == 3
    assert timing["input_tokens"] == 3 and timing["exceptions"] == 0
    assert timing["dropped_ranges"] == 0 and timing["complete"] is True
    assert [item["duration_ns"] for item in timing["ranges"]] == [0, 7, 5]
    assert [(item["start_ns"], item["end_ns"]) for item in timing["ranges"]] == [
        (10, 10),
        (100, 107),
        (1000, 1005),
    ]
    assert [item["execution_id"] for item in timing["ranges"]] == [
        "exec-1",
        "exec-2",
        None,
    ]
    assert [item["output_batch_id"] for item in timing["ranges"]] == [
        "batch-1",
        "batch-2",
        None,
    ]
    assert all(item["status"] == "ok" for item in timing["ranges"])
    assert all(item["thread_id"] == get_ident() for item in timing["ranges"])
    assert all(item["native_tid"] == get_native_id() for item in timing["ranges"])
    assert "add_request" not in record["stages"]
    assert record["official_status"] == "complete"
    official = record["official"]
    assert official["unit"] == "seconds" and official["time_to_first_token"] == 1.0
    assert official["response_time_to_first_token"] == 2.0
    assert official["first_token_observed"] is True
    assert official["e2e_latency"] == 5.0 and official["queued_time"] == 1.0
    assert official["prefill_time"] == 2.0 and official["decode_time"] == 3.0
    assert official["inference_time"] == 5.0
    assert official["mean_time_per_output_token"] == 1.5
    assert official["num_generation_tokens"] == 3
    assert stats.finished_requests[0].e2e_latency == 5.0
    assert stats.time_to_first_tokens_iter == [1.0]
    assert stats.inter_token_latencies_iter == [2.0, 1.0]
    assert request.stats.pd_batch is request.pd_batch
    assert not proc.request_states


@pytest.mark.parametrize(
    "failure,expected",
    [(ValueError("decode"), "error"), (asyncio.CancelledError(), "cancelled")],
)
def test_detokenize_failure_closes_owner_and_keeps_exception(
    api, caplog, monkeypatch, failure, expected
):
    request = state(api, log_stats=False)
    request.pd_first_output = True

    def update(*args):
        raise failure

    request.detokenizer = SimpleNamespace(update=update, num_output_tokens=lambda: 0)
    request.logprobs_processor = SimpleNamespace(update_from_output=lambda value: None)
    proc = processor(api, [request])
    ticks = iter([50, 50])
    monkeypatch.setattr(api.output, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(type(failure)) as caught:
        proc.process_outputs([output("internal", [3])])
    assert caught.value is failure
    record = records(caplog)[-1]
    assert record["status"] == expected and record["complete"] is False
    assert record["stages"]["detokenize"]["duration_ns"] == 0
    assert record["stages"]["detokenize"]["exceptions"] == 1
    timing = record["stages"]["detokenize"]
    assert timing["complete"] is True and timing["dropped_ranges"] == 0
    assert len(timing["ranges"]) == 1
    assert timing["ranges"][0]["status"] == expected
    assert timing["ranges"][0]["error_type"] == type(failure).__name__
    assert timing["ranges"][0]["execution_id"] is None
    assert timing["ranges"][0]["duration_ns"] == 0
    assert record["official_status"] == "missing"
    # Failure diagnostics do not alter business cleanup; original caller owns it.
    assert "internal" in proc.request_states


def test_detokenize_ranges_are_bounded_without_losing_cumulative_total(
    api, caplog, monkeypatch
):
    request = state(api, log_stats=False)
    request.pd_first_output = True
    request.detokenizer = SimpleNamespace(
        update=lambda *args: None, num_output_tokens=lambda: 0
    )
    request.logprobs_processor = SimpleNamespace(update_from_output=lambda value: None)
    proc = processor(api, [request])
    ticks = iter(range(140))
    monkeypatch.setattr(api.output, "monotonic_ns", lambda: next(ticks))
    with api.diag.scope(execution_id="unrelated-execution"):
        for index in range(70):
            proc.process_outputs(
                [
                    output(
                        "internal",
                        [],
                        FinishReason.STOP if index == 69 else None,
                        output_batch_id=f"batch-{index}",
                    )
                ]
            )
    timing = records(caplog)[-1]["stages"]["detokenize"]
    assert timing["duration_ns"] == 70 and timing["calls"] == 70
    assert timing["input_tokens"] == 0
    assert len(timing["ranges"]) == 64
    assert timing["dropped_ranges"] == 6 and timing["complete"] is False
    assert all(item["execution_id"] is None for item in timing["ranges"])
    assert timing["ranges"][-1]["output_batch_id"] == "batch-63"


def test_no_token_official_summary_retains_original_values(api, caplog):
    request = state(api)
    stats = api.stats.IterationStats()
    stats.iteration_timestamp = 105.0
    stats.update_from_finished_request(
        FinishReason.ABORT, "external", 2, 8, request.stats
    )
    original = stats.finished_requests[0]
    api.output.OutputProcessor._emit_request_terminal(request, "abort")
    record = records(caplog)[-1]
    assert record["official"]["first_token_observed"] is False
    assert record["official"]["time_to_first_token"] == 0.0
    assert record["official"]["queued_time"] == original.queued_time
    assert record["official"]["e2e_latency"] == original.e2e_latency


def test_executor_restores_explicit_batch_and_identity(api, monkeypatch):
    module = definitions(
        monkeypatch,
        "compact_api_test_input",
        "vllm/v1/engine/input_processor.py",
        {"InputProcessor": {"_process_inputs_with_context"}},
        diag=api.diag,
    )
    proc = module.InputProcessor()
    owner = api.diag.Batch("api_summary", scope="engine_submission")
    proc.process_inputs = lambda: (api.diag.current(), api.diag.current_batch())
    unrelated = api.diag.Batch("api_summary", scope="http")
    with api.diag.scope(request_id="wrong"), api.diag.collect(unrelated):
        context, actual = proc._process_inputs_with_context(
            {"request_id": "right"}, owner
        )
        assert context["request_id"] == "right" and actual is owner
        assert api.diag.current_batch() is unrelated
        assert api.diag.current()["request_id"] == "wrong"
    assert not owner.closed


def test_request_uses_serialized_identity_over_unrelated_context(api, caplog):
    with api.diag.scope(observation_id="ambient-wrong", submission_id="wrong"):
        request = state(
            api,
            pd_identity={
                "observation_id": "saved-http",
                "submission_id": "saved-submission",
            },
        )
    api.output.OutputProcessor._emit_request_terminal(request, "stop")
    actual = records(caplog)[-1]
    assert actual["observation_id"] == "saved-http"
    assert actual["submission_id"] == "saved-submission"
    assert actual["links"][0] == {
        "kind": "frontend",
        "observation_id": "saved-http",
        "submission_id": "saved-submission",
    }


def test_input_serializes_only_explicit_frontend_identity(api, monkeypatch):
    class PoolingParams:
        def clone(self):
            return self

    module = definitions(
        monkeypatch,
        "compact_api_test_wire",
        "vllm/v1/engine/input_processor.py",
        {"InputProcessor": {"process_inputs"}},
        diag=api.diag,
        time=time,
        SamplingParams=type("SamplingParams", (), {}),
        EngineCoreRequest=lambda **fields: SimpleNamespace(**fields),
        current_platform=SimpleNamespace(validate_request=lambda *args: None),
        split_enc_dec_input=lambda prompt: (None, prompt),
    )
    proc = module.InputProcessor()
    proc._validate_params = lambda *args: None
    proc._validate_lora = lambda *args: None
    proc._validate_model_inputs = lambda *args: None
    proc.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=True,
        )
    )
    owner = api.diag.Batch(
        "api_summary",
        scope="engine_submission",
        observation_id="http-real",
        submission_id="submission-real",
        external_req_id="external",
        non_identity="must not cross the wire",
    )
    prompt = {"type": "token", "prompt_token_ids": [1, 2], "arrival_time": 100.0}
    boundary = {**api.diag.clock_domain(), "end_ns": 99, "status": "ok"}
    prompt["pd_tokenization"] = boundary
    with api.diag.collect(owner):
        actual = proc.process_inputs("internal", prompt, PoolingParams(), ())
    assert actual.pd_identity == {
        "observation_id": "http-real",
        "submission_id": "submission-real",
        "external_req_id": "external",
    }
    assert actual.prompt_token_ids == [1, 2] and actual.arrival_time == 100.0
    assert actual.pd_tokenization is boundary
    missing = proc.process_inputs("other", prompt, PoolingParams(), ())
    assert missing.pd_identity is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_renderer_measures_tokenizer_and_marks_pretokenized_skipped(
    api, monkeypatch, caplog, asynchronous
):
    module = definitions(
        monkeypatch,
        "compact_api_test_renderer",
        "vllm/renderers/base.py",
        {
            "BaseRenderer": {
                "_tokenize_singleton_prompt",
                "_tokenize_singleton_prompt_async",
            }
        },
        diag=api.diag,
        time=time,
    )
    renderer = module.BaseRenderer()
    renderer.tokenizer = object()
    renderer._apply_prompt_char_offset = lambda prompt, offset: prompt
    renderer._tokenize_prompt = lambda prompt, params: {"prompt_token_ids": [1]}

    async def tokenize(prompt, params):
        return renderer._tokenize_prompt(prompt, params)

    renderer._tokenize_prompt_async = tokenize
    params = SimpleNamespace(
        needs_detokenization=False,
        _get_text_truncation_offset=lambda *args: 0,
        apply_pre_tokenization=lambda tokenizer, prompt: prompt,
        apply_post_tokenization=lambda tokenizer, prompt: prompt,
    )
    owner = api.diag.Batch("api_summary", scope="http")
    with api.diag.collect(owner):
        if asynchronous:
            asyncio.run(
                renderer._tokenize_singleton_prompt_async({"prompt": "hello"}, params)
            )
            asyncio.run(
                renderer._tokenize_singleton_prompt_async(
                    {"prompt_token_ids": [2]}, params
                )
            )
        else:
            renderer._tokenize_singleton_prompt({"prompt": "hello"}, params)
            renderer._tokenize_singleton_prompt({"prompt_token_ids": [2]}, params)
    owner.finish()
    stage = records(caplog)[-1]["stages"]["tokenize"]
    assert stage["calls"] == 2 and stage["measured_calls"] == 1
    assert stage["skipped_calls"] == 1 and stage["status"] == "mixed"
    assert stage["duration_ns"] >= 0


def test_concurrent_http_owners_and_disconnect_remain_independent(api, caplog):
    entered = []

    async def run():
        ready = asyncio.Event()

        async def app(scope, receive, send):
            owner = api.diag.current_batch()
            entered.append(owner.fields["observation_id"])
            if len(entered) == 2:
                ready.set()
            await ready.wait()
            assert api.diag.current_batch() is owner
            api.http.typed_handler(object(), stream=True)
            assert (await receive())["type"] == "http.disconnect"

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise AssertionError("no response expected")

        middleware = api.http.PDRequestMiddleware(app)
        await asyncio.gather(
            *[
                middleware({"type": "http", "path": "/v1/completions"}, receive, send)
                for _ in range(2)
            ]
        )
        assert api.diag.current_batch() is None

    asyncio.run(run())
    emitted = records(caplog)
    assert len(set(entered)) == 2 and len(emitted) == 2
    assert all(item["status"] == "disconnected" for item in emitted)
    assert all(not item["complete"] for item in emitted)
    assert all("first_body" not in item for item in emitted)


def test_stream_input_error_closes_only_at_original_input_endpoint(api, caplog):
    with api.engine._api_submission("external") as submission:
        with api.engine._api_submission("external", stream_input=True) as stream:
            stream.update(input_status="error", input_error_type="ValueError")
            assert not stream.closed
            stream.add_link("engine_request", request_id="last-input")
        assert not submission.closed
    stream_record, submission_record = records(caplog)
    assert stream_record["scope"] == "engine_input"
    assert stream_record["status"] == "error" and not stream_record["complete"]
    assert stream_record["error_type"] == "ValueError"
    assert stream_record["links"][-1]["request_id"] == "last-input"
    assert submission_record["status"] == "accepted"


def test_terminal_stats_failure_keeps_original_error_and_closes_request(api, caplog):
    request = state(api)
    request.pd_first_output = True
    request.detokenizer = SimpleNamespace(
        update=lambda *args: None,
        num_output_tokens=lambda: 0,
    )
    request.logprobs_processor = SimpleNamespace(update_from_output=lambda value: None)
    proc = processor(api, [request])
    failure = RuntimeError("official collector failed")

    def fail(*args):
        raise failure

    proc._update_stats_from_finished = fail
    with pytest.raises(RuntimeError) as caught:
        proc.process_outputs([output("internal", [3], FinishReason.STOP)])
    assert caught.value is failure
    actual = records(caplog)[-1]
    assert actual["status"] == "error" and actual["complete"] is False
    assert actual["error_type"] == "RuntimeError"
    assert actual["official_status"] == "unavailable"
    assert "official" not in actual
    assert request.pd_batch.closed and "internal" not in proc.request_states


def test_empty_serialized_identity_does_not_claim_ambient_http(api, caplog):
    with api.diag.scope(observation_id="unrelated", submission_id="wrong"):
        request = state(api, pd_identity={})
    api.output.OutputProcessor._emit_request_terminal(request, "stop")
    actual = records(caplog)[-1]
    assert actual["observation_id"] is None and actual["submission_id"] is None


def test_tokenizer_failure_is_original_and_retained_in_http_batch(
    api, monkeypatch, caplog
):
    module = definitions(
        monkeypatch,
        "compact_api_test_failed_renderer",
        "vllm/renderers/base.py",
        {"BaseRenderer": {"_tokenize_singleton_prompt"}},
        diag=api.diag,
    )
    renderer = module.BaseRenderer()
    renderer.tokenizer = object()
    failure = ValueError("tokenizer failed")

    def fail(*args):
        raise failure

    renderer._tokenize_prompt = fail
    params = SimpleNamespace(
        _get_text_truncation_offset=lambda *args: 0,
        apply_pre_tokenization=lambda tokenizer, prompt: prompt,
    )
    owner = api.diag.Batch("api_summary", scope="http")
    with api.diag.collect(owner):
        with pytest.raises(ValueError) as caught:
            renderer._tokenize_singleton_prompt({"prompt": "hello"}, params)
        assert caught.value is failure
        assert api.diag.current_batch() is owner and not owner.closed
    owner.finish(status="error")
    actual = records(caplog)[-1]
    assert actual["stages"]["tokenize"]["status"] == "error"
    assert actual["stages"]["tokenize"]["exceptions"] == 1
    assert actual["stages"]["tokenize"]["error_type"] == "ValueError"


@pytest.mark.parametrize("stream_input", [False, True])
def test_engine_sends_use_per_input_boundaries_without_stream_gaps(
    api, caplog, monkeypatch, stream_input
):
    now = [100]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)
    sent = []

    async def send(request):
        sent.append(request.request_id)
        now[0] += 7
        await asyncio.sleep(0)
        now[0] += 3

    engine.engine_core = SimpleNamespace(add_request_async=send)

    async def run():
        with api.engine._api_submission("external", stream_input=stream_input) as owner:
            # Input preparation or waiting for the next stream chunk is outside send.
            now[0] += 1000
            unrelated = api.diag.Batch("api_summary", scope="http")
            with api.diag.collect(unrelated):
                first = {
                    **api.diag.clock_domain(),
                    "boundary_id": "first",
                    "end_ns": now[0] - 23,
                    "status": "ok",
                }
                await engine._send_engine_request(
                    SimpleNamespace(request_id="child-0", pd_tokenization=first), owner
                )
                assert api.diag.current_batch() is unrelated
                now[0] += 2000
                second = {
                    **api.diag.clock_domain(),
                    "boundary_id": "second",
                    "end_ns": now[0] - 11,
                    "status": "ok",
                }
                await engine._send_engine_request(
                    SimpleNamespace(request_id="child-1", pd_tokenization=second), owner
                )
            now[0] += 3000
            assert "add_request" not in unrelated.stages

    asyncio.run(run())
    record = records(caplog)[-1]
    stage = record["stages"]["add_request"]
    assert sent == ["child-0", "child-1"]
    assert stage["calls"] == stage["measured_calls"] == 2
    assert stage["duration_ns"] == 54
    assert stage["start_ns"] == 1077 and stage["end_ns"] == 3120
    assert stage["status"] == "ok" and stage["exceptions"] == 0
    assert stage["source"] == "AsyncLLM.engine_core.add_request_async"
    assert stage["timing_scope"] == "tokenizer_end_to_ipc_await_return"
    assert record["scope"] == ("engine_input" if stream_input else "engine_submission")


@pytest.mark.parametrize(
    "failure,expected",
    [(ValueError("send failed"), "error"), (asyncio.CancelledError(), "cancelled")],
)
def test_engine_send_failure_preserves_exception_and_await_timing(
    api, caplog, monkeypatch, failure, expected
):
    now = [100]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)
    failures = []

    def register(*args):
        now[0] += 1000

    async def send(request):
        now[0] += 17
        await asyncio.sleep(0)
        raise failure

    def record_failure(request_id, error):
        failures.append((request_id, error))
        now[0] += 2000

    engine.engine_core = SimpleNamespace(add_request_async=send)
    engine.output_processor = SimpleNamespace(
        has_request=lambda request_id: False, add_request=register
    )
    engine.check_admission = lambda **kwargs: None
    engine._record_pd_failure = record_failure
    engine.log_requests = False

    async def run():
        with (
            pytest.raises(type(failure)) as caught,
            api.engine._api_submission("external") as owner,
        ):
            await engine._add_request(
                SimpleNamespace(
                    request_id="internal",
                    pd_tokenization={
                        **api.diag.clock_domain(),
                        "boundary_id": "failed",
                        "end_ns": 73,
                        "status": "ok",
                    },
                ),
                None,
                None,
                0,
                object(),
                owner=owner,
            )
        assert caught.value is failure

    asyncio.run(run())
    assert failures == [("internal", failure)]
    record = records(caplog)[-1]
    stage = record["stages"]["add_request"]
    assert stage["start_ns"] == 73 and stage["end_ns"] == 1117
    assert stage["duration_ns"] == 1044 and stage["calls"] == 1
    assert stage["status"] == record["status"] == expected
    assert stage["error_type"] == record["error_type"] == type(failure).__name__
    assert stage["exceptions"] == 1
    assert record["complete"] is False


def test_shared_children_charge_union_and_keep_full_send_evidence(api, monkeypatch):
    now = [100]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)

    async def send(request):
        now[0] += 10

    engine.engine_core = SimpleNamespace(add_request_async=send)
    boundary = {
        **api.diag.clock_domain(),
        "boundary_id": "shared",
        "end_ns": 20,
        "status": "ok",
    }
    owner = api.diag.Batch("api_summary")

    async def run():
        accounted = {}
        for index in range(3):
            await engine._send_engine_request(
                SimpleNamespace(request_id=f"child-{index}", pd_tokenization=boundary),
                owner,
                accounted=accounted,
            )

    asyncio.run(run())
    assert owner.stages["add_request"]["duration_ns"] == 110
    assert [link["duration_ns"] for link in owner.links] == [90, 100, 110]
    assert [link["accounted_duration_ns"] for link in owner.links] == [90, 10, 10]
    assert all(link["shared_preparation"] for link in owner.links)


@pytest.mark.parametrize("boundary_status", ["skipped", "fused", "missing", "foreign"])
def test_no_tokenizer_boundary_has_explicit_alternative_scope(
    api, monkeypatch, boundary_status
):
    now = [100]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)

    async def send(request):
        now[0] = 120

    engine.engine_core = SimpleNamespace(add_request_async=send)
    boundary = {
        **api.diag.clock_domain(),
        "boundary_id": "alternative",
        "end_ns": 40,
        "status": boundary_status,
        "reason": "test_alternative",
    }
    if boundary_status == "foreign":
        boundary["run_id"] = "another-process"
        boundary["status"] = "ok"
    owner = api.diag.Batch("api_summary")
    asyncio.run(
        engine._send_engine_request(
            SimpleNamespace(request_id="alternative", pd_tokenization=boundary), owner
        )
    )
    stage = owner.stages["add_request"]
    assert stage["timing_scope"] == "alternative_boundary_to_ipc_await_return"
    assert stage["reason"] == (
        "foreign_clock_domain" if boundary_status == "foreign" else "test_alternative"
    )
    assert stage["duration_ns"] == (20 if boundary_status == "foreign" else 80)


def test_concurrent_sends_keep_explicit_owners_and_prompt_boundaries(api, monkeypatch):
    now = [100]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)
    owners = [api.diag.Batch("api_summary", prompt_index=i) for i in range(2)]
    entered = asyncio.Event()

    async def send(request):
        if request.request_id == "first":
            entered.set()
            await asyncio.sleep(0)
            now[0] = 150
        else:
            await entered.wait()
            now[0] = 120

    engine.engine_core = SimpleNamespace(add_request_async=send)

    async def run():
        unrelated = api.diag.Batch("api_summary")
        with api.diag.collect(unrelated):
            await asyncio.gather(
                *(
                    engine._send_engine_request(
                        SimpleNamespace(
                            request_id=name,
                            pd_tokenization={
                                **api.diag.clock_domain(),
                                "boundary_id": name,
                                "end_ns": start,
                                "status": "ok",
                            },
                        ),
                        owner,
                    )
                    for name, start, owner in zip(["first", "second"], [10, 70], owners)
                )
            )
        assert unrelated.stages == {}

    asyncio.run(run())
    assert [owner.stages["add_request"]["duration_ns"] for owner in owners] == [140, 50]


def test_tokenizer_boundary_precedes_local_build_and_async_delivery(api, monkeypatch):
    now = [100]
    module = definitions(
        monkeypatch,
        "compact_boundary_renderer",
        "vllm/renderers/base.py",
        {
            "BaseRenderer": {
                "_tokenize_prompt",
                "_pd_tokenization_boundary",
                "process_for_engine",
                "process_for_engine_async",
            }
        },
        diag=api.diag,
        time=SimpleNamespace(monotonic_ns=lambda: now[0]),
    )
    renderer = module.BaseRenderer()

    def tokenize(text, **kwargs):
        now[0] = 120
        return {"input_ids": [1, 2]}

    def build(tokens, prompt, **kwargs):
        now[0] = 170
        return {"prompt_token_ids": tokens}

    renderer.get_tokenizer = lambda: tokenize
    renderer._wants_offsets = lambda *args: False
    renderer._build_tokens_prompt = build
    params = SimpleNamespace(get_encode_kwargs=lambda: {})
    prompt = renderer._tokenize_prompt({"prompt": "hello"}, params)
    assert prompt["pd_tokenization"]["end_ns"] == 120
    renderer._process_singleton = lambda *args, **kwargs: {"type": "token"}

    async def process(*args, **kwargs):
        await asyncio.sleep(0)
        now[0] = 300
        return {"type": "token"}

    renderer._process_singleton_async = process
    sync_input = renderer.process_for_engine(prompt, 1.0)
    async_input = asyncio.run(renderer.process_for_engine_async(prompt, 1.0))
    assert sync_input["pd_tokenization"] is prompt["pd_tokenization"]
    assert async_input["pd_tokenization"]["end_ns"] == 120
    assert (
        module.BaseRenderer._pd_tokenization_boundary(
            {
                "encoder_prompt": prompt,
                "decoder_prompt": {"pd_tokenization": {"status": "ok", "end_ns": 150}},
            }
        )["end_ns"]
        == 150
    )


def test_async_tokenizer_delivery_does_not_move_actual_return_boundary(
    api, monkeypatch
):
    now = [10]
    monkeypatch.setattr(api.diag, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    module = definitions(
        monkeypatch,
        "compact_async_delivery_renderer",
        "vllm/renderers/base.py",
        {"BaseRenderer": {"_tokenize_prompt", "_tokenize_singleton_prompt_async"}},
        diag=api.diag,
        time=SimpleNamespace(monotonic_ns=lambda: now[0]),
    )
    renderer = module.BaseRenderer()
    renderer.tokenizer = object()

    def tokenizer(text, **kwargs):
        now[0] = 20
        return {"input_ids": [1]}

    renderer.get_tokenizer = lambda: tokenizer
    renderer._wants_offsets = lambda *args: False
    renderer._build_tokens_prompt = lambda tokens, prompt, **kwargs: {
        "prompt_token_ids": tokens
    }

    async def deliver(prompt, params):
        result = renderer._tokenize_prompt(prompt, params)
        await asyncio.sleep(0)
        now[0] = 100
        return result

    renderer._tokenize_prompt_async = deliver
    renderer._apply_prompt_char_offset = lambda prompt, offset: prompt

    def post(tokenizer, prompt):
        now[0] = 130
        return prompt

    params = SimpleNamespace(
        needs_detokenization=False,
        _get_text_truncation_offset=lambda *args: 0,
        apply_pre_tokenization=lambda tokenizer, prompt: prompt,
        apply_post_tokenization=post,
        get_encode_kwargs=lambda: {},
    )
    engine = api.engine.AsyncLLM.__new__(api.engine.AsyncLLM)

    async def send(request):
        now[0] = 150

    engine.engine_core = SimpleNamespace(add_request_async=send)
    owner = api.diag.Batch("api_summary")

    async def run():
        prompt = await renderer._tokenize_singleton_prompt_async(
            {"prompt": "hello"}, params
        )
        assert prompt["pd_tokenization"]["end_ns"] == 20
        await engine._send_engine_request(
            SimpleNamespace(
                request_id="async", pd_tokenization=prompt["pd_tokenization"]
            ),
            owner,
        )

    asyncio.run(run())
    assert owner.stages["add_request"]["duration_ns"] == 130
