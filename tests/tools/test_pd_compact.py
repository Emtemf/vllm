# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reader-only schema-2 contracts; no serving, torch or vLLM runtime imports."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


report = module("pd_metrics_report")
export = module("pd_grafana_json")
request_export = module("pd_request_json")
api = module("grafana_rows_api")
fixture = module("pd_compact_fixture")


@pytest.fixture
def synthetic(tmp_path):
    paths = []
    for side, contents in fixture.fixture_logs().items():
        path = tmp_path / f"synthetic-{side}.log"
        path.write_text(contents, encoding="utf-8")
        paths.append(path)
    records, errors = report.read_records(paths)
    assert not errors
    return paths, records, report.summarize(records, errors)


def test_canonical_tree_keeps_request_scope_execution_and_rank(synthetic):
    _, _, result = synthetic
    dashboard = result["dashboard"]
    main = next(row for row in dashboard["requests"] if row["request_id"] == "p-main")
    assert set(main["sides"]) == {"P", "D"}
    rows = [
        row for row in dashboard["nodes"] if row["request_key"] == main["request_key"]
    ]
    for side in ("P", "D"):
        roots = [
            row for row in rows if row["side"] == side and row["parent_id"] is None
        ]
        assert {row["label"] for row in roots} == {"API Server", "EngineCore", "Worker"}
        assert all(row["duration_ns"] is None for row in roots)
    forward = [
        row for row in rows if row["side"] == "P" and row["node_key"] == "forward"
    ]
    assert {row["tp_rank"] for row in forward} == {0, 1}
    assert all(
        row["execution_id"] == "execution-P" and row["shared_batch"] for row in forward
    )
    queue = [row for row in rows if row["node_key"] == "queue"]
    assert len(queue) == 2 and all(row["scope"] == "request" for row in queue)
    assert all(row["relationship"] == "associated" for row in queue)
    assert not any(
        row["node_key"] in ("runner", "worker_summary", "logits", "total")
        for row in rows
    )
    assert not any(row["node_key"] == "req_resolve" for row in rows)
    http = next(
        row
        for row in dashboard["requests"]
        if row["observation_only"]
        and row["observation_id"] == "http-P"
        and any(source.get("scope") == "http" for source in row["source_observations"])
    )
    assert any(
        link.get("target_request_key") == main["request_key"] for link in http["links"]
    )
    assert "Observation http-P" in report.render_html(result)
    assert any(
        row["state"] == "skipped" and row["duration_ns"] is None
        for row in dashboard["nodes"]
    )
    assert any(
        row["state"] == "measured" and row["duration_ns"] == 0
        for row in dashboard["nodes"]
    )


def test_connector_timings_in_sample_summary_keep_rank_and_execution(synthetic):
    _, records, _ = synthetic
    worker = next(
        row
        for row in records
        if row.get("event") == "worker_summary" and row.get("phase") == "sample"
    )
    worker["stages"]["submit"] = {"calls": 1, "duration_ns": 123000, "status": "ok"}
    worker["stages"]["wait_for_save"] = {
        "calls": 1,
        "duration_ns": 649088,
        "status": "ok",
    }
    actual = report.summarize(records, [])
    rows = [
        row
        for row in actual["dashboard"]["nodes"]
        if any(
            source.get("record_id") == worker["record_id"]
            for source in row["source_observations"]
        )
    ]
    for name, duration in (("submit", 123000), ("wait_for_save", 649088)):
        matches = [row for row in rows if row["node_key"] == name]
        assert matches
        for row in matches:
            assert row["duration_ns"] == duration and row["state"] == "measured"
            assert row["execution_id"] == worker["execution_id"]
            assert row["tp_rank"] == worker["tp_rank"]
    assert not any(row["node_key"] == "forward" for row in rows)


def test_native_four_phases_match_without_guessing_or_fifth_total(synthetic):
    _, _, result = synthetic
    dashboard = result["dashboard"]
    native = [
        row
        for row in dashboard["nodes"]
        if row["kind"] == "native"
        and row["request_key"]
        == next(
            row["request_key"]
            for row in dashboard["requests"]
            if row["request_id"] == "p-main"
        )
    ]
    assert len(native) == 8  # One measured call and one explicit not-started call.
    assert {row["node_key"] for row in native} == set(report.NATIVE_PHASES.values())
    assert {row["call_id"] for row in native} == {"put-0", "put-2"}
    assert all(
        row["duration_ns"] is None and row["completion"] is False
        for row in native
        if row["call_id"] == "put-2"
    )
    assert (
        sum(row["duration_ns"] for row in native if row["call_id"] == "put-0") == 6000
    )
    assert any(row["kind"] == "native_unmatched" for row in dashboard["quality"])
    assert all(row["level"] == "L4" and row["l3"] == "put" for row in native)
    assert all(row["l4"] == row["node_key"] for row in native)
    assert any(row["native_total"]["duration_ns"] == 7000 for row in native)
    parents = {row["node_id"]: row for row in dashboard["nodes"]}
    assert all(parents[row["parent_id"]]["node_key"] == "put" for row in native)
    assert any(row["kind"] == "incomplete" for row in dashboard["quality"])


def test_native_precision_bucket_ambiguity_and_timezone_are_strict(synthetic):
    _, records, _ = synthetic
    native = next(row for row in records if row["event"] == "mooncake_native_timing")
    task = next(row for row in records if row.get("task_id") == "save-mixed")
    call = task["native_calls"][0]
    assert report._wall_inside(native, call)
    assert not report._wall_inside(native, {**call, "wall_utc_offset_seconds": None})
    assert not report._wall_inside(native, {**call, "wall_utc_offset_seconds": 0})
    ambiguous = {**task, "native_calls": [call, dict(call)]}
    quality = []
    assert not report._native_links([ambiguous, native], quality)
    assert quality[0]["kind"] == "native_ambiguous"
    assert quality[0]["candidate_count"] == 2
    bucket = {**native, "glog_wall": ["20261112", "08", "00", "00", "1"]}
    late = {**call, "start_wall_ns": call["start_wall_ns"] + 150_000_000}
    assert report._wall_inside(bucket, late)


def test_schema1_mixed_duplicate_and_empty_inputs_fail_closed(tmp_path, synthetic):
    paths, records, _ = synthetic
    invalid = tmp_path / "schema1.log"
    invalid.write_text(
        '[PD_DIAG] {"version":1,"event":"request_summary"}\n', encoding="utf-8"
    )
    _, errors = report.read_records([paths[0], invalid])
    with pytest.raises(ValueError, match="schema 2 required"):
        report.summarize(records, errors)
    with pytest.raises(ValueError, match="schema-2"):
        export.flatten({"version": 1})
    original = next(row for row in records if row["event"] == "api_summary")
    duplicate = tmp_path / "duplicate.log"
    clean = {k: v for k, v in original.items() if k != "log_source"}
    duplicate.write_text(
        "\n".join(
            "[PD_DIAG] " + json.dumps(row)
            for row in (clean, clean, {**clean, "status": "error"})
        ),
        encoding="utf-8",
    )
    parsed, errors = report.read_records([duplicate])
    assert len(parsed) == 1 and "conflicting duplicate" in errors[0]


def test_same_request_id_without_explicit_identity_never_merges(synthetic):
    _, records, _ = synthetic
    first = next(
        row
        for row in records
        if row["event"] == "api_summary" and row.get("scope") == "request"
    )
    copies = [
        {
            **first,
            "host": host,
            "run_id": host,
            "observation_id": None,
            "engine_id": None,
            "record_id": host,
        }
        for host in ("one", "two")
    ]
    dashboard = report.build_dashboard(copies)
    assert len(dashboard["requests"]) == 2
    assert all(row["sides"] == [None] for row in dashboard["requests"])


def test_rows_api_drilldown_official_endpoint_and_exports(synthetic):
    _, records, result = synthetic
    payload = export.flatten(result)
    assert payload["nodes"] is result["dashboard"]["nodes"]
    main = next(row for row in payload["requests"] if row["request_id"] == "p-main")
    key = main["request_key"]
    query = urlencode({"request_key": key, "side": "P", "parent_id": "__root"})
    roots = api.rows_for(payload, "/nodes?" + query)
    assert len(roots) == 3
    worker = next(row for row in roots if row["node_key"] == "worker")
    children = api.rows_for(
        payload,
        "/nodes?"
        + urlencode({"request_key": key, "side": "P", "parent_id": worker["node_id"]}),
    )
    assert children and all(row["parent_id"] == worker["node_id"] for row in children)
    p = api.rows_for(
        payload, "/requests?" + urlencode({"request_key": key, "side": "P"})
    )[0]
    d = api.rows_for(
        payload, "/requests?" + urlencode({"request_key": key, "side": "D"})
    )[0]
    assert p["ttft_ms"] is None and p["official"]["time_to_first_token"] == 0.004
    assert d["ttft_ms"] == 4 and d["e2e_ms"] == 9
    assert {
        row["side"]
        for row in api.rows_for(payload, "/sides?" + urlencode({"request_key": key}))
    } == {"P", "D"}
    links = api.rows_for(payload, "/links?" + urlencode({"request_key": key}))
    transfer = next(row for row in links if row["kind"] == "producer_consumer")
    assert transfer["source_request_key"] == transfer["target_request_key"] == key
    assert transfer["producer_engine_id"] == "engine-P"
    assert transfer["consumer_engine_id"] == "engine-D"
    other = next(row for row in payload["requests"] if row["request_id"] == "p-other")
    assert transfer not in api.rows_for(
        payload, "/links?" + urlencode({"request_key": other["request_key"]})
    )
    converted = request_export.convert(records)
    assert next(row for row in converted if row["request_key"] == key)["nodes"] == [
        row for row in payload["nodes"] if row["request_key"] == key
    ]
    receive = next(row for row in payload["pd_transfers"] if row["scope"] == "task")
    generation = next(
        row for row in payload["pd_transfers"] if row["scope"] == "request_generation"
    )
    assert receive["stages"]["wait_for_transfer"]["duration_ns"] == 20
    assert (
        receive["transfer_envelope_ns"] == 999
        and generation["transfer_envelope_ns"] == 1999
    )


def test_actual_cli_outputs_and_failure_preserves_existing_file(synthetic, tmp_path):
    paths, _, _ = synthetic
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "pd_metrics_report.py"),
            *map(str, paths),
            "-o",
            str(output),
            "--html",
        ],
        check=True,
        capture_output=True,
    )
    assert output.with_suffix(".html").exists()
    rows = tmp_path / "rows.json"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "pd_grafana_json.py"),
            str(output),
            "-o",
            str(rows),
        ],
        check=True,
        capture_output=True,
    )
    api.validate_payload(json.loads(rows.read_text(encoding="utf-8")))
    request_output = tmp_path / "requests.json"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "pd_request_json.py"),
            *map(str, paths),
            "-o",
            str(request_output),
        ],
        check=True,
        capture_output=True,
    )
    assert all(
        row["version"] == 2
        for row in json.loads(request_output.read_text(encoding="utf-8"))
    )
    empty = tmp_path / "empty.log"
    empty.write_text("no diagnostics\n", encoding="utf-8")
    previous = rows.read_bytes()
    failed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "pd_grafana_json.py"),
            "--logs",
            str(empty),
            "-o",
            str(rows),
        ],
        capture_output=True,
    )
    assert failed.returncode == 2 and rows.read_bytes() == previous


def test_native_duplicate_samples_and_incomplete_without_durations(synthetic):
    _, records, _ = synthetic
    task = next(row for row in records if row.get("task_id") == "save-mixed")
    native = next(row for row in records if row["event"] == "mooncake_native_timing")
    quality = []
    assert not report._native_links([task, native, dict(native)], quality)
    assert any(row["kind"] == "native_ambiguous" for row in quality)
    incomplete = report._native_timing(
        "I20261112 08:00:00.000000 77 store.cpp:1] [BatchPutTiming] "
        "timing_scope=normal_full status=error complete=false phase_status=not_started"
    )
    assert incomplete["duration_fields_ns"] == {}
    with pytest.raises(ValueError, match="four phases"):
        report._native_timing(
            "[BatchPutTiming] timing_scope=normal_full "
            "status=ok complete=true total_us=1"
        )


def test_position_observations_and_failed_zero_measurement_are_preserved(synthetic):
    _, _, result = synthetic
    positions = [
        row for row in result["dashboard"]["observations"] if row.get("kind") == "event"
    ]
    assert len(positions) == 2
    assert all(
        row["duration_ns"] is None and row["state"] == "event" for row in positions
    )
    failed = next(
        row
        for row in result["dashboard"]["nodes"]
        if row.get("task_id") == "save-truncated" and row["node_key"] == "put"
    )
    assert failed["state"] == "measured" and failed["duration_ns"] == 0
    assert failed["status"] == "error" and failed["exceptions"] == 1


def test_request_ranges_keep_execution_and_unassigned_calls_separate(synthetic):
    _, records, _ = synthetic
    core = next(
        row
        for row in records
        if row["event"] == "core_summary" and row.get("scope") == "request"
    )
    core["stages"]["hash"] = {
        "calls": 3,
        "duration_ns": 60,
        "status": "ok",
        "complete": True,
        "dropped_ranges": 0,
        "ranges": [
            {"duration_ns": 10, "status": "ok", "execution_id": None},
            {"duration_ns": 20, "status": "ok", "execution_id": "execution-P"},
            {"duration_ns": 30, "status": "ok", "execution_id": "execution-next"},
        ],
    }
    core["stages"]["queue"] = {
        "calls": 1,
        "duration_ns": 100,
        "status": "ok",
        "ranges": [
            {
                "start_ns": 1,
                "end_ns": 101,
                "duration_ns": 100,
                "execution_id": "execution-P",
                "status": "ok",
            }
        ],
    }
    dashboard = report.summarize(records, [])["dashboard"]
    rows = [
        n for n in dashboard["nodes"] if n.get("record_index") == records.index(core)
    ]
    hashes = [n for n in rows if n["node_key"] == "hash"]
    assert {(n["execution_id"], n["duration_ns"]) for n in hashes} == {
        (None, 10),
        ("execution-P", 20),
        ("execution-next", 30),
    }
    queue = next(n for n in rows if n["node_key"] == "queue")
    assert queue["execution_id"] == "execution-P"
    assert queue["duration_ns"] == 100 and queue["relationship"] == "associated"
    assert all(n["duration_ns"] != 60 for n in hashes)


def test_unused_pool_stages_remain_evidence_not_business_nodes(synthetic):
    _, records, _ = synthetic
    task = next(row for row in records if row.get("task_id") == "save-mixed")
    task["stages"]["sync"] = {"calls": 0, "status": "unmeasured"}
    dashboard = report.summarize(records, [])["dashboard"]
    assert not any(
        n.get("task_id") == "save-mixed" and n["node_key"] == "sync"
        for n in dashboard["nodes"]
    )
    evidence = next(
        o
        for o in dashboard["observations"]
        if o.get("task_id") == "save-mixed" and o.get("field") == "sync"
    )
    assert evidence["value"] == {"calls": 0, "status": "unmeasured"}
