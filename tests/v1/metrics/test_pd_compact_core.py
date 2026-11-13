# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run actual core instrumentation without importing torch or device runtimes."""

import ast
import asyncio
import enum
import importlib.util
import itertools
import json
import logging
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "compact_core_diagnostics", ROOT / "vllm/v1/metrics/pd_diagnostics.py"
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def compile_subject(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            if isinstance(node, ast.ClassDef) and methods is not None:
                node.body = [
                    part
                    for part in node.body
                    if isinstance(part, ast.FunctionDef) and part.name in methods
                ]
            body.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *body,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace


@pytest.fixture
def subject(caplog):
    caplog.set_level(logging.INFO, logger=diag.__name__)
    ticks = itertools.count()
    clock = SimpleNamespace(monotonic_ns=lambda: next(ticks), time=lambda: 0)
    namespace = dict(
        diag=diag,
        time=clock,
        enum=enum,
        deque=deque,
        wraps=wraps,
        contextmanager=contextmanager,
        Future=Future,
        cast=cast,
        Any=Any,
        ModelRunnerOutput=object,
        SchedulerInterface=object,
        StructuredOutputRequest=SimpleNamespace(from_sampling_params=lambda _: None),
        PrefillStats=SimpleNamespace,
        ConstantList=lambda values: values,
        length_from_prompt_token_ids_or_embeds=lambda tokens, embeds: len(tokens),
    )
    compile_subject("vllm/v1/request.py", {"Request", "RequestStatus"}, namespace)
    compile_subject("vllm/v1/core/kv_cache_manager.py", {"_pd_range"}, namespace)
    compile_subject(
        "vllm/v1/core/sched/scheduler.py", {"Scheduler"}, namespace, {"schedule"}
    )
    compile_subject(
        "vllm/v1/engine/core.py",
        {"EngineCore"},
        namespace,
        {
            "_pd_schedule",
            "_pd_execution_scope",
            "_pd_attach_output_identity",
            "_enqueue_output",
            "step",
            "step_with_batch_queue",
            "shutdown",
        },
    )
    return SimpleNamespace(**namespace)


def rows(caplog):
    return [
        json.loads(record.getMessage().split("[PD_DIAG] ", 1)[1])
        for record in caplog.records
        if "[PD_DIAG] " in record.getMessage()
    ]


def request(subject, request_id="A", hasher=None, identity=None):
    params = SimpleNamespace(
        max_tokens=4, extra_args=None, skip_reading_prefix_cache=None
    )
    return subject.Request(
        request_id, [1, 2], params, None, block_hasher=hasher, pd_identity=identity
    )


def test_request_queue_hash_are_terminal_once_and_not_ambient(subject, caplog):
    value = request(subject, hasher=lambda _: [b"initial"])
    value.pd_identity = {"engine_id": "engine-A"}
    value.pd_queue_transition("queued")
    value.pd_execution_id = "exec-A"
    value.pd_queue_transition("scheduled")
    value.pd_executions += 1
    with diag.scope(request_ids=["A"], execution_id="exec-A"):
        value.update_block_hashes()
    value.pd_queue_transition("preempted")
    value.status = subject.RequestStatus.FINISHED_ABORTED
    with (
        diag.scope(
            request_id="B",
            request_ids=["B"],
            observation_id="api-B",
            execution_id="exec-B",
        ),
        diag.batch("core_summary"),
    ):
        value.pd_queue_transition("finished")
        value.pd_queue_transition("finished")
    records = [row for row in rows(caplog) if row.get("scope") == "request"]
    assert len(records) == 1
    row = records[0]
    assert (row["request_id"], row["engine_id"], row["execution_id"]) == (
        "A",
        "engine-A",
        None,
    )
    assert row["observation_id"] is None
    assert row["status"] == "cancelled"
    queue = row["stages"]["queue"]
    assert queue["calls"] == 2
    assert queue["duration_ns"] == sum(r["duration_ns"] for r in queue["ranges"])
    assert [r["execution_id"] for r in queue["ranges"]] == ["exec-A", None]
    hashed = row["stages"]["hash"]
    assert hashed["calls"] == 2
    assert [r["execution_id"] for r in hashed["ranges"]] == [None, "exec-A"]
    assert hashed["duration_ns"] == sum(r["duration_ns"] for r in hashed["ranges"])


def test_unrelated_execution_does_not_own_new_request_hash(subject, caplog):
    with diag.scope(request_ids=["B"], execution_id="exec-B"):
        value = request(subject, hasher=lambda _: [])
    value.status = subject.RequestStatus.FINISHED_STOPPED
    value.pd_queue_transition("finished")
    hashed = rows(caplog)[0]["stages"]["hash"]
    assert hashed["ranges"][0]["execution_id"] is None


def test_explicit_frontend_identity_survives_initial_hash_and_terminal(subject, caplog):
    identity = {"observation_id": "actual-api-observation", "prompt_index": 2}
    value = request(subject, hasher=lambda _: [], identity=identity)
    assert value.pd_identity == identity
    assert value.pd_identity is not identity
    value.status = subject.RequestStatus.FINISHED_STOPPED
    value.pd_queue_transition("finished")
    row = rows(caplog)[0]
    assert row["observation_id"] == identity["observation_id"]
    assert row["members"][0]["observation_id"] == identity["observation_id"]
    assert row["prompt_index"] == 2


def test_engine_request_conversion_copies_explicit_identity(subject):
    original = request(subject, identity={"observation_id": "frontend-owner"})
    payload = SimpleNamespace(
        **vars(original), reasoning_ended=None, reasoning_parser_kwargs=None
    )
    converted = subject.Request.from_engine_core_request(payload, None)
    assert converted.pd_identity == {"observation_id": "frontend-owner"}
    assert converted.pd_identity is not original.pd_identity


def test_request_bounded_evidence_marks_loss_incomplete(subject, caplog):
    value = request(subject, hasher=lambda _: [])
    for _ in range(70):
        value.pd_queue_transition("queued")
        value.pd_queue_transition("scheduled")
        value.update_block_hashes()
    value.status = subject.RequestStatus.FINISHED_STOPPED
    value.pd_queue_transition("finished")
    row = rows(caplog)[0]
    assert row["complete"] is False
    assert row["stages"]["queue"]["dropped_ranges"] == 6
    assert row["stages"]["hash"]["dropped_ranges"] == 7
    assert len(row["stages"]["hash"]["ranges"]) == 64
    assert row["stages"]["hash"]["calls"] == 71


@pytest.mark.parametrize("failure", [ValueError("hasher"), asyncio.CancelledError()])
def test_hash_failure_keeps_original_exception_and_emits_once(subject, caplog, failure):
    value = request(subject)

    def fail(_):
        raise failure

    value._block_hasher = fail
    with pytest.raises(type(failure)) as caught:
        value.update_block_hashes()
    assert caught.value is failure
    value.pd_queue_transition("finished")
    assert len(rows(caplog)) == 1
    row = rows(caplog)[0]
    expected = "cancelled" if isinstance(failure, asyncio.CancelledError) else "error"
    assert row["status"] == expected
    assert row["complete"] is False
    assert row["stages"]["hash"]["exceptions"] == 1


def execution(request_id, pending=False, tokens=1):
    return SimpleNamespace(
        pd_execution_id=None,
        num_scheduled_tokens={request_id: tokens},
        total_num_scheduled_tokens=tokens,
        pending_structured_output_tokens=pending,
    )


def resolved(value=None, failure=None):
    future = Future()
    if failure is None:
        future.set_result(value)
    else:
        future.set_exception(failure)
    return future


def engine(subject, outputs, *, queue_size=3, execute=None, sample=None):
    pending = deque(outputs)
    core = subject.EngineCore.__new__(subject.EngineCore)
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        kv_transfer_config=None,
    )
    core.scheduler = SimpleNamespace(
        _pd_engine_id="engine",
        has_requests=lambda: bool(pending),
        schedule=lambda _: pending.popleft(),
        get_grammar_bitmask=lambda output: output,
        update_from_output=lambda output, result: {
            0: SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        request_id=request_id,
                        pd_execution_id=None,
                        output_batch_id=None,
                    )
                    for request_id in output.num_scheduled_tokens
                ],
                output_batch_id=None,
                finished_requests=None,
            )
        },
    )
    core.model_executor = SimpleNamespace(
        execute_model=execute or (lambda output, **_: resolved(object())),
        sample_tokens=sample or (lambda output, **_: resolved(object())),
    )
    core.batch_queue_size = queue_size
    core.batch_queue = deque(maxlen=queue_size)
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.check_for_draft_tokens = False
    core._should_throttle_prefills = lambda: False
    core.log_error_detail = lambda _: nullcontext()
    core.capture_iteration_details = lambda _: nullcontext(None)
    core._process_aborts_queue = lambda: None
    core._attach_iteration_details = lambda *_: None
    core._attach_weight_version = lambda *_: None
    return core


def test_batch_queue_owns_each_summary_through_submit_pop_and_drain(subject, caplog):
    first, second = execution("A"), execution("B")
    observed = []

    def sample(output, **_):
        observed.append((diag.current()["execution_id"], diag.current_batch()))
        return resolved(object())

    core = engine(subject, [first, second], sample=sample)
    assert core.step_with_batch_queue() == (None, True)
    assert core.step_with_batch_queue() == (None, True)
    assert len(core.batch_queue[0]) == 4
    assert rows(caplog) == []
    with diag.scope(request_id="unrelated", execution_id="ambient"):
        first_outputs = core.step_with_batch_queue()[0][0]
        second_outputs = core.step_with_batch_queue()[0][0]
        assert first_outputs.outputs[0].pd_execution_id == first.pd_execution_id
        assert second_outputs.outputs[0].pd_execution_id == second.pd_execution_id
        assert first_outputs.output_batch_id != second_outputs.output_batch_id
        assert first_outputs.outputs[0].output_batch_id == first_outputs.output_batch_id
        assert (
            second_outputs.outputs[0].output_batch_id == second_outputs.output_batch_id
        )
        assert diag.current()["execution_id"] == "ambient"
    records = rows(caplog)
    assert [row["execution_id"] for row in records] == [
        first.pd_execution_id,
        second.pd_execution_id,
    ]
    assert [row["request_ids"] for row in records] == [["A"], ["B"]]
    assert all(row["request_id"] is None for row in records)
    assert [identity for identity, _ in observed] == [
        first.pd_execution_id,
        second.pd_execution_id,
    ]
    assert observed[0][1] is not observed[1][1]
    assert core.step_with_batch_queue() == (None, False)


def test_deferred_sample_preserves_new_execution_after_old_result(subject, caplog):
    first, deferred = execution("A"), execution("B", pending=True)
    sampled = []

    def sample(output, **_):
        sampled.append(diag.current()["execution_id"])
        return resolved(object())

    core = engine(subject, [first, deferred], sample=sample)
    core.step_with_batch_queue()
    completed = core.step_with_batch_queue()[0][0]
    assert completed.outputs[0].pd_execution_id == first.pd_execution_id
    assert core.batch_queue[0][1] is deferred
    core.step_with_batch_queue()
    assert sampled == [first.pd_execution_id, deferred.pd_execution_id]
    assert [row["execution_id"] for row in rows(caplog)] == sampled


@pytest.mark.parametrize("failure", [ValueError("execute"), asyncio.CancelledError()])
def test_queued_execute_failure_preserved_and_all_summaries_closed(
    subject,
    caplog,
    failure,
):
    first, second = execution("A"), execution("B")
    core = engine(
        subject,
        [first, second],
        queue_size=2,
        execute=lambda *_args, **_kwargs: resolved(failure=failure),
        sample=lambda *_args, **_kwargs: resolved(None),
    )
    core.step_with_batch_queue()
    with pytest.raises(type(failure)) as caught:
        core.step_with_batch_queue()
    assert caught.value is failure
    records = rows(caplog)
    assert len(records) == 2
    assert {row["execution_id"] for row in records} == {
        first.pd_execution_id,
        second.pd_execution_id,
    }
    status = "cancelled" if isinstance(failure, asyncio.CancelledError) else "error"
    assert all(row["status"] == status and row["complete"] is False for row in records)
    assert all(item[3].closed for item in core.batch_queue)


def test_sync_execution_finishes_one_summary_after_sample(subject, caplog):
    output = execution("A")
    core = engine(
        subject,
        [output],
        execute=lambda *_args, **_kwargs: resolved(None),
        sample=lambda *_args, **_kwargs: object(),
    )
    outputs, executed = core.step()
    assert executed is True
    assert outputs[0].outputs[0].pd_execution_id == output.pd_execution_id
    assert outputs[0].outputs[0].output_batch_id == outputs[0].output_batch_id
    assert len(rows(caplog)) == 1
    assert rows(caplog)[0]["execution_id"] == output.pd_execution_id


def test_shutdown_closes_queued_owner_without_waiting_on_future(subject, caplog):
    output = execution("A")
    pending = Future()
    core = engine(
        subject,
        [output],
        execute=lambda *_args, **_kwargs: pending,
        sample=lambda *_args, **_kwargs: pending,
    )
    core.step_with_batch_queue()
    core.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
    core.model_executor.shutdown = lambda: None
    core.scheduler.shutdown = lambda: None
    core.shutdown.__func__.__globals__.update(
        logger=SimpleNamespace(debug_once=lambda *_: None),
        gc=SimpleNamespace(unfreeze=lambda: None),
        cleanup_dist_env_and_memory=lambda: None,
    )
    core.shutdown()
    assert pending.done() is False
    assert len(rows(caplog)) == 1
    assert rows(caplog)[0]["completion"] == "shutdown"
    assert rows(caplog)[0]["status"] == "cancelled"


@pytest.mark.parametrize("queued", [False, True])
def test_execution_identity_reaches_all_clients_and_empty_error_outputs(
    subject, queued
):
    scheduled = execution("A")
    scheduled.num_scheduled_tokens["failed"] = 1
    core = engine(subject, [scheduled], queue_size=1)
    first = SimpleNamespace(request_id="A", pd_execution_id=None, output_batch_id=None)
    second = SimpleNamespace(request_id="A", pd_execution_id=None, output_batch_id=None)
    failed = SimpleNamespace(
        request_id="failed",
        new_token_ids=[],
        finish_reason="error",
        pd_execution_id=None,
        output_batch_id=None,
    )
    unrelated_error = SimpleNamespace(
        request_id="grammar-error", pd_execution_id=None, output_batch_id=None
    )
    batches = {
        0: SimpleNamespace(
            outputs=[first, second, unrelated_error],
            output_batch_id=None,
            finished_requests=None,
        ),
        1: SimpleNamespace(
            outputs=[failed], output_batch_id=None, finished_requests={"failed"}
        ),
        2: SimpleNamespace(
            outputs=[], output_batch_id=None, finished_requests={"finished"}
        ),
    }
    core.scheduler.update_from_output = lambda *_: batches
    with diag.scope(execution_id="unrelated"):
        actual, _ = core.step_with_batch_queue() if queued else core.step()
    assert actual is batches
    assert len({batch.output_batch_id for batch in batches.values()}) == 3
    for batch in batches.values():
        assert batch.output_batch_id is not None
        for member in batch.outputs:
            assert member.pd_execution_id == (
                scheduled.pd_execution_id
                if member.request_id in scheduled.num_scheduled_tokens
                else None
            )
            assert member.output_batch_id == batch.output_batch_id


def test_execution_without_request_outputs_does_not_fabricate_output(subject):
    core = engine(subject, [execution("A")])
    core.scheduler.update_from_output = lambda *_: {}
    assert core.step() == ({}, True)


def test_preexecution_error_transport_keeps_execution_unknown(subject):
    core = subject.EngineCore.__new__(subject.EngineCore)
    core.output_queue = SimpleNamespace(put_nowait=lambda value: sent.append(value))
    sent = []
    member = SimpleNamespace(pd_execution_id=None, output_batch_id=None)
    batch = SimpleNamespace(
        outputs=[member],
        finished_requests={"A"},
        utility_output=None,
        output_batch_id=None,
    )
    with diag.scope(execution_id="unrelated"):
        core._enqueue_output(0, batch)
    assert sent == [(0, batch)]
    assert member.pd_execution_id is None
    assert member.output_batch_id == batch.output_batch_id
    assert batch.output_batch_id is not None


def external_lookup_subject(subject):
    tree = ast.parse(
        (ROOT / "vllm/v1/core/sched/scheduler.py").read_text(encoding="utf-8")
    )
    # Extract the real connector call and its finally block, including timing inputs.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_schedule":
            continue
        for block in ast.walk(node):
            if not isinstance(block, ast.If):
                continue
            for index, statement in enumerate(block.body):
                if isinstance(
                    statement, ast.Try
                ) and "get_num_new_matched_tokens" in ast.unparse(statement):
                    function = ast.FunctionDef(
                        name="external_lookup",
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[
                                ast.arg(arg=name)
                                for name in ("self", "request", "block_aligned_local")
                            ],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=block.body[index - 3 : index + 1],
                        decorator_list=[],
                    )
                    namespace = {"time": subject.time, "diag": diag}
                    module = ast.Module(body=[function], type_ignores=[])
                    exec(
                        compile(
                            ast.fix_missing_locations(module), "external_lookup", "exec"
                        ),
                        namespace,
                    )
                    return namespace["external_lookup"]
    raise AssertionError("Scheduler external lookup instrumentation was removed")


def scheduler(subject, callback):
    value = subject.Scheduler.__new__(subject.Scheduler)
    value._pd_engine_id = "engine"
    value.vllm_config = SimpleNamespace(kv_transfer_config=None)
    value.parallel_config = SimpleNamespace(data_parallel_index=0)
    value.kv_cache_manager = SimpleNamespace(pd_stage_ns={})
    value._schedule = lambda throttle: callback(value)
    return value


def test_schedule_nonoverlapping_local_external_alloc_and_other(subject, caplog):
    observe = subject._pd_range

    class Manager:
        def __init__(self):
            self._pd_range_depth = 0
            self.pd_stage_ns = {}

        @observe("lookup")
        def lookup(self):
            self.nested_lookup()

        @observe("lookup")
        def nested_lookup(self):
            return None

        @observe("alloc")
        def allocate(self):
            return object()

        @observe("free")
        def free(self):
            return None

    external = external_lookup_subject(subject)

    def schedule(value):
        value.kv_cache_manager.lookup()
        external(value, None, 0)
        value.kv_cache_manager.allocate()
        value.kv_cache_manager.free()
        return execution("A")

    value = scheduler(subject, schedule)
    manager = value.kv_cache_manager = Manager()
    value.connector = SimpleNamespace(get_num_new_matched_tokens=lambda *_: (0, False))
    value.schedule()
    row = rows(caplog)[0]
    stages = row["stages"]
    assert set(stages) == {"schedule", "lookup", "alloc", "other"}
    assert stages["lookup"]["calls"] == 2  # one outer local call + external
    assert stages["lookup"]["duration_ns"] == (
        manager.pd_stage_ns["lookup"] + value._pd_external_lookup_ns
    )
    assert stages["other"]["duration_ns"] == (
        stages["schedule"]["duration_ns"]
        - sum(manager.pd_stage_ns.values())
        - value._pd_external_lookup_ns
    )
    assert row["request_ids"] == ["A"]


def test_empty_schedule_preserves_skipped_stages(subject, caplog):
    output = execution("A", tokens=0)
    output.num_scheduled_tokens = {}
    scheduler(subject, lambda _: output).schedule()
    row = rows(caplog)[0]
    assert row["no_forward"] is True
    for name in ("lookup", "alloc"):
        assert row["stages"][name]["status"] == "skipped"
        assert row["stages"][name]["duration_ns"] is None


@pytest.mark.parametrize("failure", [ValueError("schedule"), asyncio.CancelledError()])
def test_schedule_failure_finishes_once_without_replacing_exception(
    subject,
    caplog,
    failure,
):
    def fail(_):
        raise failure

    value = scheduler(subject, fail)
    owner = diag.Batch("core_summary", scope="execution")
    with diag.collect(owner), pytest.raises(type(failure)) as caught:
        value.schedule()
    assert caught.value is failure
    owner.finish()
    assert len(rows(caplog)) == 1
    row = rows(caplog)[0]
    assert row["complete"] is False
    assert row["stages"]["schedule"]["exceptions"] == 1
