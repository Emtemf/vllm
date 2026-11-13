#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write deterministic SYNTHETIC schema-2 P/D logs for CLI and UI smoke checks."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def fixture_logs() -> dict[str, str]:
    logs = {"P": [], "D": []}
    count = 0
    wall = int(datetime(2026, 11, 12, 0, 0, tzinfo=UTC).timestamp()) * 10**9

    def stage(duration=None, calls=1, status="ok", **fields):
        return {
            "calls": calls,
            "status": status,
            "duration_ns": duration,
            "exceptions": int(status == "error"),
            **fields,
        }

    def record(side, event, stages=None, **fields):
        nonlocal count
        count += 1
        value = {
            "version": 2,
            "event": event,
            "host": "synthetic-" + side,
            "pid": 100 if side == "P" else 200,
            "native_tid": 77,
            "run_id": "SYNTHETIC-" + side,
            "timestamp_ns": count * 1000,
            "record_id": "synthetic-" + str(count),
            "synthetic": True,
            "side": side,
            "status": "ok",
            "complete": True,
            "stages": stages or {},
            **fields,
        }
        value.pop("side", None)
        logs[side].append("[PD_DIAG] " + json.dumps(value, sort_keys=True))
        return value

    def native(second, tid=77, keys=2, scope="normal_full", complete=True):
        fields = (
            f"timing_scope={scope} status={'ok' if complete else 'error'} "
            f"complete={str(complete).lower()} keys={keys} bytes=128"
        )
        if scope == "prefer_same_node_not_started":
            fields += " phase_status=not_started"
        else:
            fields += (
                " phase1_start_batch_put_us=1 phase2_submit_transfers_us=2"
                " phase3_wait_transfers_us=3 phase4_finalize_batch_put_us=0 total_us=7"
            )
        logs["P"].append(
            f"I20261112 08:00:{second:02d}.100000 {tid} store.cpp:42] [BatchPutTiming] "
            + fields
        )

    for side, rid in (("P", "p-main"), ("D", "d-main")):
        observation, engine = "http-" + side, "engine-" + side
        record(
            side,
            "api_summary",
            {
                "req_resolve": stage(10),
                "render": stage(0),
                "tokenize": stage(None, 0, "skipped"),
            },
            scope="http",
            observation_id=observation,
            links=[
                {"kind": "engine_submission", "submission_id": "submission-" + side}
            ],
            first_body={"timestamp_ns": 999, "source": "http"},
        )
        record(
            side,
            "api_summary",
            {"add_request": stage(None, 0, "missing")},
            scope="engine_submission",
            observation_id=observation,
            submission_id="submission-" + side,
            links=[{"kind": "engine_request", "request_id": rid}],
        )
        record(
            side,
            "api_summary",
            {
                "add_request": stage(None, 0, "missing"),
                "detokenize": stage(30, 3, timing_scope="cumulative"),
            },
            scope="request",
            observation_id=observation,
            request_id=rid,
            submission_id="submission-" + side,
            official={
                "unit": "seconds",
                "time_to_first_token": 0.004,
                "e2e_latency": 0.009,
                "decode_time": 0.002,
                "first_token_observed": side == "D",
                "num_prompt_tokens": 8,
                "num_generation_tokens": 1 if side == "D" else 0,
            },
        )
        members = [{"request_id": rid, "observation_id": observation}]
        if side == "P":
            members.append({"request_id": "p-other", "observation_id": "http-other"})
        record(
            side,
            "core_summary",
            {
                "schedule": stage(100),
                "lookup": stage(20, 2),
                "alloc": stage(0),
                "other": stage(50),
            },
            scope="execution",
            engine_id=engine,
            execution_id="execution-" + side,
            members=members,
        )
        record(
            side,
            "core_summary",
            {"queue": stage(6, 2, timing_scope="cumulative"), "hash": stage(0)},
            scope="request",
            request_id=rid,
            engine_id=engine,
            observation_id=observation,
        )
        for rank in (0, 1):
            record(
                side,
                "worker_summary",
                {
                    "submit": stage(10),
                    "forward": stage(80 + rank),
                    "wait_for_save": stage(None, 0, "skipped"),
                },
                phase="execute",
                execution_id="execution-" + side,
                tp_rank=rank,
                request_ids=[member["request_id"] for member in members],
            )
            record(
                side,
                "worker_summary",
                {"sampling": stage(0)},
                phase="sample",
                execution_id="execution-" + side,
                tp_rank=rank,
                request_ids=[member["request_id"] for member in members],
            )
    calls = []
    for number in range(4):
        calls.append(
            {
                "call_id": f"put-{number}",
                "native_tid": 77,
                "start_wall_ns": wall + number * 10**9,
                "end_wall_ns": wall + number * 10**9 + 500_000_000,
                "wall_utc_offset_seconds": 8 * 3600,
                "duration_ns": 500_000_000,
                "calls": 1,
                "status": "ok",
                "submitted_keys": 2,
                "submitted_bytes": 128,
            }
        )
    pool_stages = {
        name: stage(10, 2) for name in ("build_key", "lookup", "prepare", "sync")
    }
    pool_stages["put"] = stage(2_000_000_000, 4)
    record(
        "P",
        "pool_task",
        pool_stages,
        operation="save",
        scope="task",
        task_id="save-mixed",
        execution_id="execution-P",
        tp_rank=0,
        members=[
            {"request_id": "p-main", "execution_id": "execution-P"},
            {"request_id": "p-other", "execution_id": "execution-P"},
        ],
        native_calls=calls,
    )
    native(0)
    native(1, keys=999)  # Count mismatch cannot claim the second call.
    native(2, scope="prefer_same_node_not_started", complete=False)
    native(3, tid=999)  # Thread mismatch cannot claim the fourth call.
    # Early native failure retains incomplete evidence without phase durations.
    logs["P"].append(
        "I20261112 08:00:09.100000 77 store.cpp:42] [BatchPutTiming] "
        "timing_scope=normal_full status=error complete=false phase_status=not_started"
    )
    record(
        "D",
        "pd_transfer_summary",
        {"submit": stage(10, 2), "wait_for_transfer": stage(20, 2)},
        scope="task",
        direction="read",
        request_id="d-main",
        engine_id="engine-D",
        remote_request_id="p-main",
        remote_engine_id="engine-P",
        task_id="receive-1",
        request_generation=1,
        transfer_envelope_ns=999,
    )
    record(
        "D",
        "pd_transfer_summary",
        scope="request_generation",
        direction="read",
        request_id="d-main",
        engine_id="engine-D",
        request_generation=1,
        transfer_envelope_ns=1999,
    )
    record(
        "P",
        "api_summary",
        {"req_resolve": stage(5, status="error")},
        scope="http",
        observation_id="http-failed",
        status="error",
        complete=False,
        completion="exception",
    )
    record(
        "P",
        "pool_task",
        {"put": stage(0, status="error")},
        operation="save",
        scope="task",
        request_id="p-other",
        execution_id="execution-P",
        task_id="save-truncated",
        native_calls=[],
        native_calls_dropped=2,
        status="error",
        complete=False,
    )
    return {
        side: "SYNTHETIC compact schema-2 fixture; not performance evidence\n"
        + "\n".join(lines)
        + "\n"
        for side, lines in logs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for side, contents in fixture_logs().items():
        (args.out / f"synthetic-{side}.log").write_text(contents, encoding="utf-8")
    print(f"SYNTHETIC P/D fixtures -> {args.out}")


if __name__ == "__main__":
    main()
