# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.engine.async_llm as async_llm_module
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics import pd_diagnostics as diag


@pytest.mark.asyncio
@pytest.mark.parametrize("log_stats", [False, True])
@pytest.mark.parametrize("has_batch_id", [False, True])
@pytest.mark.parametrize("num_outputs", [0, 3])
async def test_output_batch_boundaries(
    monkeypatch, caplog, log_stats, has_batch_id, num_outputs
):
    """Boundaries describe one received batch, even across chunk yields."""
    monkeypatch.setattr(async_llm_module.envs, "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", 1)
    caplog.set_level(logging.INFO, logger=diag.__name__)
    queue = asyncio.Queue()
    drained = asyncio.Event()
    get_calls = 0

    async def get_output_async():
        nonlocal get_calls
        get_calls += 1
        if get_calls == 2:
            drained.set()
        return await queue.get()

    def observations():
        return [
            json.loads(record.getMessage().split("[PD_DIAG] ", 1)[1])
            for record in caplog.records
            if record.name == diag.__name__
        ]

    processed = []

    def process_outputs(outputs_slice, timestamp, iteration_stats, **kwargs):
        # Both boundaries must precede processing, including the first chunk.
        events = observations()
        assert events[0]["event"] == "api_output_received"
        if log_stats:
            assert events[1]["event"] == "api_output_endpoint"
            assert events[1]["iteration_timestamp"] == iteration_stats.iteration_timestamp
        else:
            assert iteration_stats is None
            assert len(events) == 1
        processed.extend(outputs_slice)
        return SimpleNamespace(request_outputs=[], reqs_to_abort=[])

    processor = SimpleNamespace(
        process_outputs=Mock(side_effect=process_outputs),
        update_scheduler_stats=Mock(),
        propagate_error=Mock(),
    )
    logger = Mock()
    llm = SimpleNamespace(
        output_handler=None,
        engine_core=SimpleNamespace(get_output_async=get_output_async),
        output_processor=processor,
        log_stats=log_stats,
        logger_manager=logger,
        renderer=SimpleNamespace(mm_processor_cache=None, stat_mm_cache=lambda: None),
    )
    batch = SimpleNamespace(
        outputs=[SimpleNamespace(request_id=f"req-{i}") for i in range(num_outputs)],
        finished_requests={"finished-only", "req-0"} if num_outputs else None,
        timestamp=17.0,
        weight_version=None,
        engine_index=0,
        scheduler_stats=None,
    )
    if has_batch_id:
        batch.output_batch_id = "batch-1"

    # A background task inherits the launching request's context. None of that
    # request's identity may be attributed to this later, unrelated batch.
    identity_fields = (
        "request_id", "observation_id", "external_req_id", "parent_request_id",
        "api_request_id", "prompt_index",
    )
    with diag.scope(**dict.fromkeys(identity_fields, "launching-request")):
        AsyncLLM._run_output_handler(llm)
    try:
        await asyncio.sleep(0)
        assert observations() == []  # Empty-queue waiting is not a batch timer.
        queue.put_nowait(batch)
        await asyncio.wait_for(drained.wait(), timeout=5)
    finally:
        llm.output_handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await llm.output_handler

    processor.propagate_error.assert_not_called()
    assert processed == batch.outputs
    assert processor.process_outputs.call_count == num_outputs
    events = observations()
    assert [event["event"] for event in events] == (
        ["api_output_received", "api_output_endpoint"]
        if log_stats and num_outputs else ["api_output_received"]
    )
    assert isinstance(events[0]["received_ns"], int)
    assert events[0]["received_ns"] <= events[0]["timestamp_ns"]
    expected_ids = [f"req-{i}" for i in range(num_outputs)]
    if num_outputs:
        expected_ids.append("finished-only")
    for event in events:
        assert event["request_ids"] == expected_ids
        assert event["output_batch_id"] == ("batch-1" if has_batch_id else None)
        assert event["batch_shared"] is True
        assert event["execution_id"] is None
        assert all(event[field] is None for field in identity_fields)
        assert isinstance(event["timestamp_ns"], int)
        assert "duration_ns" not in event
    recorded_stats = logger.record.call_args.kwargs["iteration_stats"]
    if log_stats and num_outputs:
        assert recorded_stats is not None
        assert events[1]["iteration_timestamp"] == recorded_stats.iteration_timestamp
    else:
        assert recorded_stats is None
