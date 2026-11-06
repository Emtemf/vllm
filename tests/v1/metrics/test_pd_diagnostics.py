# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib.util
import json
import logging
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

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


@pytest.fixture
def report_module():
    """The offline tool must be importable without initializing an engine."""
    path = Path(__file__).resolve().parents[3] / "tools" / "pd_metrics_report.py"
    spec = importlib.util.spec_from_file_location("pd_metrics_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_record(event, **fields):
    return {
        "version": 1, "event": event, "host": "P", "pid": 10, "run_id": "core",
        "native_tid": 1, **fields,
    }


def transport_records(request_id="external-a1b2c3d4"):
    frontend = {"pid": 1, "run_id": "frontend", "observation_id": "api-unique"}
    return [
        report_record("request_id_link", **frontend, request_id=request_id,
                      external_req_id="external"),
        report_record("engine_submit", **frontend, request_id=request_id,
                      start_ns=10, end_ns=20),
        report_record("engine_core_received", request_id=request_id),
    ]


def test_report_links_unique_randomized_transport_and_keeps_tp_separate(report_module):
    records = transport_records() + [
        report_record("scheduler_execution", execution_id="E",
                      request_ids=["external-a1b2c3d4"],
                      members=[{"request_id": "external-a1b2c3d4",
                                "scheduled_tokens": 7, "num_output_tokens": 0}]),
        report_record("scheduler_schedule", execution_id="E",
                      request_ids=["external-a1b2c3d4"], start_ns=30, end_ns=40,
                      manager_ranges_ns={"lookup": 0}, other_ns=10),
        *[report_record("runner.forward", execution_id="E", tp_rank=rank,
                        pid=20 + rank, run_id=f"worker-{rank}",
                        request_ids=["external-a1b2c3d4"], start_ns=100, end_ns=200)
          for rank in (0, 1)],
    ]
    dashboard = report_module.summarize(records, [])["dashboard"]
    assert len(dashboard["requests"]) == 1
    assert len(dashboard["transport_links"]) == 1
    execution = dashboard["executions"][0]
    assert len(execution["workers"]) == 2
    assert execution["members"][0]["scheduled_tokens"] == 7
    assert all(stage["event"] != "scheduler_schedule"
               for worker in execution["workers"] for stage in worker["stages"])
    forward = next(stage for stage in dashboard["requests"][0]["overview"]
                   if stage["event"] == "runner.forward")
    assert [m["duration_ns"] for m in forward["measurements"]] == [100, 100]
    assert "total_ns" not in forward


def test_report_reused_internal_id_across_lifecycles_is_only_candidate(report_module):
    records = transport_records() + [
        report_record("engine_core_received", request_id="external-a1b2c3d4",
                      host="another-host", pid=10, run_id="restarted"),
    ]
    dashboard = report_module.build_dashboard(records)
    assert not dashboard["transport_links"]
    assert len(dashboard["requests"]) == 3
    candidate = dashboard["transport_candidates"][0]
    assert len(candidate["receiver_keys"]) == 2
    sender = next(r for r in dashboard["requests"]
                  if r["key"] == candidate["sender_keys"][0])
    assert not sender["execution_ids"]


def test_report_reused_external_id_does_not_merge_api_observations(report_module):
    records = [
        report_record("request_id_link", run_id="frontend", observation_id=observation,
                      external_req_id="reused", request_id="reused")
        for observation in ("api-first", "api-second")
    ]
    assert len(report_module.build_dashboard(records)["requests"]) == 2


def test_report_mixed_pool_task_does_not_merge_or_split_members(report_module):
    records = [
        report_record("scheduler_execution", execution_id="E", request_ids=["A", "B"],
                      members=[{"request_id": "A", "scheduled_tokens": 8},
                               {"request_id": "B", "scheduled_tokens": 1}]),
        report_record("pool_task", pid=20, run_id="worker", execution_id=None,
                      task_id="pool-mixed", request_ids=["A", "B"], tp_rank=0,
                      operation="save", start_ns=100, end_ns=200,
                      stages={"prepare": {"duration_ns": 80, "calls": 1},
                              "backend_put": {"duration_ns": 50, "calls": 1}}),
        *[report_record("pool_task_member", pid=20, run_id="worker",
                        execution_id="E", task_id="pool-mixed", request_id=member,
                        operation="save") for member in ("A", "B")],
    ]
    dashboard = report_module.build_dashboard(records)
    assert len(dashboard["requests"]) == 2
    task = dashboard["tasks"][0]
    assert len(task["request_keys"]) == 2
    assert len(task["envelopes"]) == 1
    assert task["envelopes"][0]["duration_ns"] == 100
    assert [stage["duration_ns"] for stage in task["stages"]] == [80, 50]
    assert "total_ns" not in task


def test_report_invalid_clock_missing_and_zero_are_distinct(report_module):
    valid = report_record("forward", start_ns=2, end_ns=2)
    assert report_module._measurement([valid])["duration_ns"] == 0
    invalid = {**valid, "start_ns": 9, "end_ns": 2, "duration_ns": 0}
    assert report_module._measurement([invalid])["state"] == "missing"
    unknown_clock = {**valid, "run_id": None}
    assert report_module._measurement([unknown_clock])["state"] == "missing"
    skipped = report_record("tokenize", status="skipped", duration_ns=None,
                            reason="pretokenized_or_embeds")
    assert report_module._measurement([skipped])["state"] == "not_applicable"
    assert report_module._measurement([])["state"] == "missing"
    overlapping = [report_record("forward", start_ns=a, end_ns=b)
                   for a, b in ((10, 20), (15, 30))]
    assert report_module._measurement(overlapping)["duration_ns"] == 20


def test_report_worker_hierarchy_requires_same_native_thread(report_module):
    records = [
        report_record("runner.forward", start_ns=2, end_ns=8),
        report_record("runner.execute", start_ns=1, end_ns=10),
        report_record("runner.sampling", native_tid=2, start_ns=3, end_ns=4),
    ]
    labels = {r["event"]: r["event"] for r in records}
    rows = report_module._worker_stages(records, labels, ("P", 10, "core", 0, 0, 0))
    forward = next(row for row in rows if row["event"] == "runner.forward")
    sample = next(row for row in rows if row["event"] == "runner.sampling")
    assert forward["parent_event"] == "runner.execute"
    assert forward["depth"] == 1
    assert sample["parent_event"] is None


def test_report_native_no_token_and_html_escape_boundary(report_module):
    hostile = '</script><img src=x onerror="alert(1)">'
    records = [
        report_record("request_summary", request_id=hostile, first_token_observed=False,
                      time_to_first_token=0.01, num_generation_tokens=0, status="error"),
        report_record("tokenize", request_id=hostile, status="skipped",
                      reason="pretokenized_or_embeds", duration_ns=None),
    ]
    report = report_module.summarize(records, [])
    assert report["schema_version"] == 1
    assert report["dashboard"]["requests"][0]["native_summaries"][0][
        "display_ms"]["time_to_first_token"] == 10
    page = report_module.render_html(report)
    assert hostile not in page
    assert "&lt;/script&gt;&lt;img" in page
    assert page.count("<script>") == 1


def test_report_empty_and_p_only_do_not_require_decode_logs(report_module):
    assert report_module.summarize([], [])["dashboard"]["requests"] == []
    records = transport_records() + [
        report_record("pd_transfer_offer", request_id="external-a1b2c3d4",
                      transfer_engine_id="P-engine", transfer_request_id="external-a1b2c3d4",
                      transfer_role="producer"),
    ]
    dashboard = report_module.build_dashboard(records)
    assert len(dashboard["requests"]) == 1
    assert dashboard["requests"][0]["roles"] == ["producer"]
    assert not dashboard["unresolved_links"]


def test_report_nonrandomized_transport_is_unconfirmed_even_with_unique_receiver(report_module):
    dashboard = report_module.build_dashboard(transport_records("external"))
    assert not dashboard["transport_links"]
    assert len(dashboard["requests"]) == 2
    assert len(dashboard["transport_candidates"]) == 1


def test_report_mixed_token_inputs_preserve_measured_and_skipped_states(report_module):
    records = [
        report_record("tokenize", observation_id="api", status="skipped",
                      reason="pretokenized_or_embeds", duration_ns=None),
        report_record("tokenize", observation_id="api", status="ok",
                      start_ns=10, end_ns=20),
    ]
    overview = report_module.build_dashboard(records)["requests"][0]["overview"]
    tokenize = next(stage for stage in overview if stage["event"] == "tokenize")
    assert [m["state"] for m in tokenize["measurements"]] == [
        "not_applicable", "measured"]
    assert tokenize["measurements"][1]["duration_ns"] == 10


def test_report_worker_only_membership_and_hash_original_execution(report_module):
    records = [
        report_record("request_hash_ranges", request_id="A", complete=True,
                      ranges=[{"execution_id": "E", "start_ns": 2, "end_ns": 4}]),
        report_record("scheduler_execution", execution_id="E", request_ids=["A"]),
        report_record("runner.forward", pid=20, run_id="worker", execution_id="W",
                      request_ids=["B"], tp_rank=0, start_ns=10, end_ns=30),
    ]
    dashboard = report_module.build_dashboard(records)
    assert len(dashboard["requests"]) == 2
    execution = dashboard["executions"][0]
    hashes = [stage for stage in execution["shared"]
              if stage["event"] == "request_hash_call"]
    assert hashes[0]["duration_ns"] == 2
    worker_only = dashboard["executions"][1]
    assert worker_only["anchor_source"] == "worker_execution_membership_only"
    assert len(worker_only["request_keys"]) == 1
    assert worker_only["members"] == [{"request_id": "B"}]


def test_report_partial_pool_ranks_reuse_unique_worker_domain(report_module):
    records = [
        report_record("runner.forward", execution_id="E", request_ids=["A"],
                      tp_rank=0, dp_rank=0, pp_rank=0, start_ns=10, end_ns=20),
        report_record("pool_task", execution_id="E", request_id="A",
                      tp_rank=0, task_id="pool", operation="save",
                      start_ns=20, end_ns=30),
    ]
    workers = report_module.build_dashboard(records)["executions"][0]["workers"]
    assert len(workers) == 1
    assert workers[0]["domain"] == ["P", 10, "core", 0, 0, 0]


@pytest.mark.parametrize("different", [
    {"dp_rank": 1, "pp_rank": 0},
    {"dp_rank": 0, "pp_rank": 1},
    {"dp_rank": 0, "pp_rank": 0, "run_id": "another-lifecycle"},
])
def test_report_partial_ranks_do_not_choose_conflicting_or_other_lifecycle(
    report_module, different
):
    records = [
        report_record("runner.forward", execution_id="E", request_ids=["A"],
                      tp_rank=0, dp_rank=0, pp_rank=0, start_ns=10, end_ns=20),
        report_record("runner.logits", execution_id="E", request_ids=["A"],
                      tp_rank=0, start_ns=20, end_ns=30, **different),
        report_record("pool_task", execution_id="E", request_id="A",
                      tp_rank=0, task_id="pool", operation="save",
                      start_ns=30, end_ns=40,
                      **({"run_id": "unknown-lifecycle"}
                         if "run_id" in different else {})),
    ]
    workers = report_module.build_dashboard(records)["executions"][0]["workers"]
    assert len(workers) == 3
    assert any(worker["domain"][3:5] == [None, None] for worker in workers)


def sheet_rows(page):
    class SheetParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.row = self.cell = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "tr" and "class" in attrs:
                self.row = {"kind": attrs["class"], "cells": {}}
            if tag == "td" and self.row is not None:
                self.cell = (attrs.get("data-stage", str(len(self.row["cells"]))), [])

        def handle_data(self, data):
            if self.cell is not None:
                self.cell[1].append(data)

        def handle_endtag(self, tag):
            if tag == "td" and self.cell is not None:
                name, text = self.cell
                self.row["cells"][name] = "".join(text)
                self.cell = None
            if tag == "tr" and self.row is not None:
                self.rows.append(self.row)
                self.row = None

    parser = SheetParser()
    parser.feed(page)
    return parser.rows


def test_report_scheduler_notes_do_not_reuse_previous_task_state(report_module):
    records = [
        report_record("scheduler_execution", execution_id=step, request_ids=[request],
                      members=[{"request_id": request, "scheduled_tokens": tokens}])
        for step, request, tokens in (("E1", "A", 17), ("E2", "B", 23))
    ] + [
        report_record("pool_task", execution_id="E1", task_id="idle", status="no_task",
                      operation="save_submit", start_ns=1, end_ns=2),
    ]
    rows = sheet_rows(report_module.render_html(report_module.summarize(records, [])))
    shared = [row["cells"] for row in rows if row["kind"] == "shared-row"]
    assert list(shared[0].values())[-1] == "A: 17 tokens"
    assert list(shared[1].values())[-1] == "B: 23 tokens"


def test_report_orphan_backend_and_pd_batch_remain_visible_without_fake_step(report_module):
    records = [
        report_record("pool_task", task_id="orphan-save", request_id="A",
                      operation="save", start_ns=0, end_ns=4_000_000,
                      stages={"backend_put": {"duration_ns": 3_000_000}}),
        report_record("pd_transfer_batch_finished", task_id="pd", request_id="B",
                      operation="read", start_ns=5_000_000, end_ns=7_000_000),
        report_record("pd_transfer_task_finished", task_id="pd", request_id="B",
                      operation="read", start_ns=4_000_000, end_ns=8_000_000),
    ]
    rows = sheet_rows(report_module.render_html(report_module.summarize(records, [])))
    tasks = [r["cells"] for r in rows if r["kind"] == "task-row"]
    assert tasks[0]["1"] == "未关联"
    assert tasks[0]["backend_put"] == "3"
    assert tasks[1]["pd_transfer_batch_finished"] == "2"
    assert tasks[1]["pd_transfer_task_finished"] == "4"


def test_report_completion_identity_is_not_scheduled_compute_membership(report_module):
    records = [
        report_record("engine_core_received", request_id="A"),
        report_record("scheduler_execution", execution_id="finish", request_ids=[],
                      no_forward=True, members=[]),
        report_record("kv.completion_observed", execution_id="finish", pid=20,
                      run_id="worker", tp_rank=0, request_ids=[],
                      finished_sending=["A"], finished_recving=[], failed_recving=[]),
    ]
    dashboard = report_module.build_dashboard(records)
    execution = dashboard["executions"][0]
    assert execution["request_keys"] == [dashboard["requests"][0]["key"]]
    assert execution["members"] == []
    assert execution["completion_request_ids"] == ["A"]


def test_report_native_phase1_and_remaining_keep_separate_scopes(report_module, tmp_path):
    path = tmp_path / "native.log"
    path.write_text(
        "[BatchPutTiming][prefer_same_node] keys=4 phase1_start_batch_put_us=100 "
        "timing_scope=phase1_only\n"
        "[BatchPutTiming][prefer_same_node] keys=4 phase2_submit_transfers_us=20 "
        "phase3_wait_transfers_us=30 phase4_finalize_batch_put_us=10 total_us=60 "
        "timing_scope=prefer_same_node_remaining\n"
        "[BatchPutTiming] phase1_start_batch_put_us=-1\n",
        encoding="utf-8",
    )
    records, errors = report_module.read_records([path])
    assert len(errors) == 1
    report = report_module.summarize(records, errors)
    assert report["dashboard"]["requests"] == []
    native = report["native_timings"]
    assert native[0]["duration_fields_ns"] == {"phase1_start_batch_put": 100_000}
    assert native[1]["duration_fields_ns"] == {
        "phase2_submit_transfers": 20_000, "phase3_wait_transfers": 30_000,
        "phase4_finalize_batch_put": 10_000, "total": 60_000,
    }
    assert all(link["task_key"] is None for link in report["dashboard"]["native_links"])


def test_report_hash_update_external_lookup_and_unassociated_fence_are_not_lost(
    report_module,
):
    records = [
        report_record("request_hash_ranges", request_id="A", complete=True,
                      ranges=[{"start_ns": 0, "end_ns": 1_000_000}]),
        report_record("scheduler_execution", execution_id="E", request_ids=["A"]),
        report_record("scheduler_schedule", execution_id="E", request_ids=["A"],
                      start_ns=0, end_ns=10_000_000, external_lookup_ns=5_000_000),
        report_record("engine_scheduler_update", execution_id="E",
                      start_ns=10_000_000, end_ns=13_000_000),
        report_record("pool_save_fence", request_id=None, execution_id=None,
                      pid=20, run_id="worker", request_ids=["A"], tp_rank=0,
                      start_ns=20_000_000, end_ns=22_000_000),
    ]
    rows = sheet_rows(report_module.render_html(report_module.summarize(records, [])))
    request = next(r["cells"] for r in rows if r["kind"] == "request-row")
    shared = next(r["cells"] for r in rows if r["kind"] == "shared-row")
    fence = next(r["cells"] for r in rows if r["kind"] == "fence-row")
    assert request["request_hash_call"] == "1"
    assert shared["external_lookup"] == "5"
    assert shared["engine_scheduler_update"] == "3"
    assert fence["pool_save_fence"] == "2"
    assert fence["0"] == "未关联"


def test_report_native_tid_recovers_worker_but_not_two_requests_on_one_thread(
    report_module, tmp_path,
):
    path = tmp_path / "prefill.log"
    tasks = [
        report_record("pool_task", request_id=request, task_id=task,
                      operation="save", pid=20, run_id="worker", native_tid=101,
                      tp_rank=3, dp_rank=0, status="succeeded",
                      start_ns=start, end_ns=start + 2_000_000)
        for request, task, start in (("A", "save-A", 0), ("B", "save-B", 3_000_000))
    ]
    path.write_text(
        "\n".join("[PD_DIAG] " + json.dumps(record) for record in tasks)
        + "\nI20261106 12:34:56.123456 101 client_service.cpp:3420] "
        "[BatchPutTiming] keys=4 phase1_start_batch_put_us=100 "
        "phase2_submit_transfers_us=20 total_us=120 timing_scope=normal_full\n",
        encoding="utf-8",
    )
    records, errors = report_module.read_records([path])
    assert errors == []
    report = report_module.summarize(records, errors)
    link = report["dashboard"]["native_links"][0]
    assert link["thread_owner"]["domain"] == ["P", 20, "worker"]
    assert link["thread_owner"]["tp_rank"] == 3
    assert link["thread_owner"]["task_candidates"] == ["task-1", "task-2"]
    assert link["task_key"] is None
    assert all(not any("phase" in stage["event"] for stage in task["stages"])
               for task in report["dashboard"]["tasks"])


@pytest.mark.parametrize("different", ["run_id", "host", "file"])
def test_report_native_tid_does_not_cross_lifecycles_or_log_sources(
    report_module, tmp_path, different,
):
    path = tmp_path / "prefill.log"
    worker = report_record("pool_task", request_id="A", task_id="save-A",
                           operation="save", native_tid=101, tp_rank=0)
    other = {**worker, different: "other"}
    path.write_text(
        "[PD_DIAG] " + json.dumps(worker) + "\n"
        + ("[PD_DIAG] " + json.dumps(other) + "\n" if different != "file" else "")
        + "I1106 12:34:56.123456 101 client_service.cpp:3420] "
        "[BatchPutTiming] total_us=120\n",
        encoding="utf-8",
    )
    if different == "file":
        native_path = tmp_path / "native.log"
        native_path.write_text(path.read_text(encoding="utf-8").splitlines()[-1] + "\n",
                               encoding="utf-8")
        path.write_text("[PD_DIAG] " + json.dumps(worker) + "\n", encoding="utf-8")
        paths = [path, native_path]
    else:
        paths = [path]
    records, errors = report_module.read_records(paths)
    assert errors == []
    link = report_module.summarize(records, errors)["dashboard"]["native_links"][0]
    assert link["thread_owner"] is None
    assert link["association"] == (
        "missing_thread_owner" if different == "file" else "ambiguous_thread_owner")


def test_report_worker_coverage_uses_union_not_nested_wrapper_durations(report_module):
    records = [
        report_record("scheduler_execution", execution_id="E", request_ids=["A"])
    ]
    for event, start, end, tid in (
        ("worker.execute", 0, 100, 1),
        ("runner.execute", 5, 95, 1),
        ("runner.forward", 10, 70, 1),
        ("runner.logits", 60, 80, 1),
        ("kv.save_wait", 0, 100, 2),
    ):
        records.append(
            report_record(
                event,
                execution_id="E",
                request_ids=["A"],
                tp_rank=0,
                native_tid=tid,
                start_ns=start,
                end_ns=end,
                status="ok",
            )
        )
    coverage = report_module.build_dashboard(records)["executions"][0]["workers"][0][
        "coverage"
    ]
    assert coverage["execute"] == {
        "duration_ns": 100,
        "covered_ns": 70,
        "uncovered_ns": 30,
        "state": "measured",
    }
    assert coverage["sample"]["duration_ns"] is None


def test_report_invalid_child_coverage_preserves_outer_duration(report_module):
    records = [
        report_record(
            "worker.execute",
            execution_id="E",
            request_ids=["A"],
            tp_rank=0,
            start_ns=0,
            end_ns=100,
            status="ok",
        ),
        report_record(
            "runner.forward",
            execution_id="E",
            request_ids=["A"],
            tp_rank=0,
            start_ns=90,
            end_ns=110,
            status="ok",
        ),
    ]
    coverage = report_module.build_dashboard(records)["executions"][0]["workers"][0][
        "coverage"
    ]["execute"]
    assert coverage["duration_ns"] == 100
    assert coverage["covered_ns"] is None
    assert coverage["uncovered_ns"] is None


@pytest.mark.parametrize("date", ["1106", "20261106"])
def test_report_native_wall_brackets_separate_requests_on_the_same_thread(
    report_module,
    tmp_path,
    date,
):
    start = (
        int(datetime(2026, 11, 6, 4, 34, 56, tzinfo=timezone.utc).timestamp()) * 10**9
    )
    records = []
    for request, task, offset in (("A", "save-A", 0), ("B", "save-B", 200_000_000)):
        records.append(
            report_record(
                "pool_native_call",
                request_id=request,
                task_id=task,
                operation="mooncake_put",
                tp_rank=0,
                start_wall_ns=start + offset,
                end_wall_ns=start + offset + 100_000_000,
                wall_utc_offset_seconds=8 * 3600,
                submitted_keys=4,
                submitted_bytes=1024,
            )
        )
        records.append(
            report_record(
                "pool_task",
                request_id=request,
                task_id=task,
                operation="save",
                tp_rank=0,
                start_ns=offset,
                end_ns=offset + 100_000_000,
                counters={"put_submitted_keys": 4, "put_submitted_bytes": 1024},
            )
        )
    path = tmp_path / "prefill.log"
    path.write_text(
        "\n".join("[PD_DIAG] " + json.dumps(record) for record in records)
        + f"\nI{date} 12:34:56.050000 1 client_service.cpp:1] "
        "[BatchPutTiming] keys=4 bytes=1024 phase1_start_batch_put_us=100 total_us=100\n"
        + f"I{date} 12:34:56.250000 1 client_service.cpp:1] "
        "[BatchPutTiming] keys=4 bytes=1024 phase1_start_batch_put_us=200 total_us=200\n",
        encoding="utf-8",
    )
    parsed, errors = report_module.read_records([path])
    assert errors == []
    report = report_module.summarize(parsed, errors)
    assert [link["task_key"] for link in report["dashboard"]["native_links"]] == [
        "task-1",
        "task-2",
    ]
    rows = {
        row["cells"]["0"]: row["cells"]
        for row in sheet_rows(report_module.render_html(report))
        if row["kind"] == "task-row"
    }
    assert rows["A"]["phase1_start_batch_put"] == "0.1"
    assert rows["B"]["phase1_start_batch_put"] == "0.2"


@pytest.mark.parametrize(
    "mismatch", ["overlap", "precision", "keys", "bytes", "tid", "file"]
)
def test_report_native_does_not_assign_ambiguous_or_inconsistent_calls(
    report_module,
    tmp_path,
    mismatch,
):
    start = (
        int(datetime(2026, 11, 6, 4, 34, 56, tzinfo=timezone.utc).timestamp()) * 10**9
    )
    call = report_record(
        "pool_native_call",
        request_id="A",
        task_id="save-A",
        operation="mooncake_put",
        start_wall_ns=start,
        end_wall_ns=start + 100_000_000,
        wall_utc_offset_seconds=8 * 3600,
        submitted_keys=4,
        submitted_bytes=1024,
    )
    if mismatch == "keys":
        call["submitted_keys"] = 8
    if mismatch == "bytes":
        call["submitted_bytes"] = 2048
    if mismatch == "tid":
        call["native_tid"] = 2
    records = [call]
    if mismatch == "overlap":
        records.append({**call, "task_id": "save-B", "request_id": "B"})
    if mismatch == "precision":
        call["end_wall_ns"] = start + 50_000_500
        records.append(
            {
                **call,
                "task_id": "save-B",
                "request_id": "B",
                "start_wall_ns": start + 50_000_500,
                "end_wall_ns": start + 100_000_000,
            }
        )
    native = (
        "I1106 12:34:56.050000 1 client_service.cpp:1] "
        "[BatchPutTiming] keys=4 bytes=1024 total_us=100\n"
    )
    path = tmp_path / "prefill.log"
    path.write_text(
        "\n".join("[PD_DIAG] " + json.dumps(r) for r in records)
        + "\n"
        + ("" if mismatch == "file" else native),
        encoding="utf-8",
    )
    paths = [path]
    if mismatch == "file":
        native_path = tmp_path / "native.log"
        native_path.write_text(native, encoding="utf-8")
        paths.append(native_path)
    parsed, errors = report_module.read_records(paths)
    assert errors == []
    link = report_module.summarize(parsed, errors)["dashboard"]["native_links"][0]
    assert link["task_key"] is None


def test_report_shares_scheduler_cells_without_copying_frontend_work_to_tp(
    report_module,
):
    records = [
        report_record("scheduler_execution", execution_id="E", request_ids=["A"]),
        report_record(
            "scheduler_schedule",
            execution_id="E",
            request_ids=["A"],
            start_ns=0,
            end_ns=10_000_000,
        ),
        report_record(
            "frontend_request_terminal",
            request_id="A",
            detokenize_calls=1,
            detokenize_duration_ns=7_000_000,
        ),
    ]
    for tp, end in ((0, 40_000_000), (1, 50_000_000)):
        records.append(
            report_record(
                "runner.forward",
                execution_id="E",
                request_ids=["A"],
                pid=20 + tp,
                run_id=f"worker-{tp}",
                tp_rank=tp,
                start_ns=20_000_000,
                end_ns=end,
            )
        )
    page = report_module.render_html(report_module.summarize(records, []))
    rows = sheet_rows(page)
    workers = [row["cells"] for row in rows if row["kind"] == "worker-row"]
    assert workers[0]["scheduler_schedule"] == "10"
    assert "scheduler_schedule" not in workers[1]
    assert workers[0]["runner.forward"] == "20"
    assert workers[1]["runner.forward"] == "30"
    assert all(row["detokenize_duration_ns"] == "—" for row in workers)
    request = next(row["cells"] for row in rows if row["kind"] == "request-row")
    assert request["detokenize_duration_ns"] == "7"

    class LeafOrder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fields = []

        def handle_starttag(self, tag, attrs):
            field = dict(attrs).get("data-stage")
            if tag == "th" and field:
                self.fields.append(field)

    headers = LeafOrder()
    headers.feed(page)
    assert headers.fields.index("runner.sampling") < headers.fields.index(
        "detokenize_duration_ns"
    )
