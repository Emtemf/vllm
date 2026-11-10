#!/usr/bin/env python3
"""Convert PD_DIAG logs to request-scoped JSON using report associations.

The output preserves observed execution, TP, and pool-task measurements.
Fields without an input observation stay null and are listed in ``missing``.
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pd_metrics_report import build_dashboard, read_records

WORKER_EVENTS = {
    "kv.load_submit": "start_load_kv",
    "runner.input_prepare": "input_prepare",
    "runner.forward": "forward",
    "runner.logits": "logits",
    "kv.finish_forward": "finish_forward",
    "runner.sampling": "sampling",
    "runner.draft": "draft",
    "runner.output_prepare": "output_prepare",
    "kv.save_wait": "wait_for_put",
}


class Missing(dict):
    """No special type; keeps the public schema represented as normal JSON."""


def milliseconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value / 1_000_000, 3)
    return None


def wall_timestamp(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        wall = record.get("glog_wall")
        if not wall:
            continue
        year = int(wall[0]) if len(wall[0]) == 4 else 2000 + int(wall[0])
        value = datetime.strptime(
            f"{year}{wall[1]}{wall[2]}{wall[3]}{wall[4]}",
            "%Y%m%d%H%M%S%f",
        ).replace(tzinfo=UTC)
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return None


def request_role(request: dict[str, Any]) -> str | None:
    roles = set(request.get("roles", []))
    if roles & {"kv_producer", "prefill", "producer", "P"}:
        return "p"
    if roles & {"kv_consumer", "decode", "consumer", "D"}:
        return "d"
    return None


def stage_observation(measurement: dict[str, Any], event: str) -> dict[str, Any]:
    duration = measurement.get("duration_ns")
    return {
        "event": event,
        "state": measurement.get("state", "missing"),
        "ms": milliseconds(duration),
        "execution_id": measurement.get("execution_id"),
        "domain": measurement.get("domain"),
        "note": measurement.get("note"),
    }


def add_stage(
    target: dict[str, Any], name: str, observations: list[dict[str, Any]], level: int
):
    if not observations:
        return
    measured = [row for row in observations if row["state"] == "measured"]
    value = {"level": level, "ms": measured[0]["ms"] if len(measured) == 1 else None}
    value["observations"] = observations
    target[name] = value


def task_json(
    task: dict[str, Any], all_records: list[dict[str, Any]]
) -> dict[str, Any]:
    task_records = [all_records[index] for index in task["record_indices"]]
    result = {
        "task_id": task["task_id"],
        "operation": task["operation"],
        "status": task["statuses"],
        "execution_ids": task["execution_ids"],
        "tp": None,
        "counters": task["counters"],
        "stages": {},
        "envelopes": task["envelopes"],
    }
    ranks = {
        record.get("tp_rank")
        for record in task_records
        if record.get("tp_rank") is not None
    }
    result["tp"] = next(iter(ranks)) if len(ranks) == 1 else None
    for record in task_records:
        for name, stage in record.get("stages", {}).items():
            result["stages"].setdefault(name, []).append(
                {
                    "state": "measured"
                    if stage.get("duration_ns") is not None
                    else "missing",
                    "ms": milliseconds(stage.get("duration_ns")),
                    "calls": stage.get("calls"),
                    "exceptions": stage.get("exceptions"),
                }
            )
    return result


def convert(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dashboard = build_dashboard(records)
    executions = {row["execution_id"]: row for row in dashboard["executions"]}
    tasks = {row["key"]: row for row in dashboard["tasks"]}
    output = []

    for request in dashboard["requests"]:
        request_records = [records[index] for index in request["record_indices"]]
        phase = request_role(request)
        missing = ["model", "tokens.prompt", "tokens.output", "latency_ms.e2e"]
        if phase is None:
            missing.append("route.role")

        ttft = {"p": {}, "d": {}}
        selected_phase = ttft[phase] if phase else None
        for overview in request["overview"]:
            event = overview["event"]
            if event == "scheduler_schedule":
                stages = [
                    stage
                    for stage in overview["measurements"]
                    if stage.get("state") != "missing"
                ]
                observations = [stage_observation(stage, event) for stage in stages]
                if selected_phase is not None:
                    add_stage(selected_phase, "schedule", observations, 1)
                    for execution_id in request["execution_ids"]:
                        execution = executions.get(execution_id)
                        if execution is None:
                            continue
                        matching = [
                            records[index]
                            for index in execution["record_indices"]
                            if records[index].get("event") == "scheduler_schedule"
                        ]
                        if len(matching) != 1:
                            continue
                        ranges = matching[0].get("manager_ranges_ns", {})
                        sub = selected_phase.setdefault("schedule_sub", {})
                        for source, name in (
                            ("queue", "queue_time"),
                            ("hash", "hash"),
                            ("lookup", "lookup"),
                            ("alloc", "alloc"),
                            ("other", "other"),
                        ):
                            if source in ranges:
                                sub.setdefault(name, []).append(
                                    {
                                        "execution_id": execution_id,
                                        "ms": milliseconds(ranges[source]),
                                    }
                                )
            elif event in WORKER_EVENTS:
                observations = [
                    stage_observation(stage, event)
                    for stage in overview["measurements"]
                    if stage.get("state") != "missing"
                ]
                if selected_phase is not None:
                    by_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for observation in observations:
                        domain = observation.get("domain") or []
                        rank = str(domain[-1]) if domain else "unknown"
                        by_rank[rank].append(observation)
                    name = WORKER_EVENTS[event]
                    for rank, rank_observations in by_rank.items():
                        worker = selected_phase.setdefault("worker_tp", {}).setdefault(
                            rank, {}
                        )
                        add_stage(worker, name, rank_observations, 1)

        task_rows = [tasks[key] for key in request["task_keys"] if key in tasks]
        save_tasks = [
            task_json(task, records)
            for task in task_rows
            if "save" in task["operation"]
        ]
        background = {"p": {}}
        if save_tasks:
            by_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for task in save_tasks:
                by_rank[
                    str(task["tp"]) if task["tp"] is not None else "unknown"
                ].append(task)
            background["p"]["wait_for_save"] = {
                "level": 1,
                "tp": [
                    {"rank": rank, "tasks": grouped}
                    for rank, grouped in sorted(by_rank.items())
                ],
            }

        api_terminal = next(
            (
                row
                for row in request_records
                if row["event"] in ("api_terminal", "frontend_request_terminal")
            ),
            {},
        )
        summaries = [
            row for row in request_records if row["event"] == "request_summary"
        ]
        metrics = request["metrics"]
        official = summaries[0] if len(summaries) == 1 else {}
        official_ms = {
            name: official.get(name) * 1000
            if isinstance(official.get(name), (int, float))
            else None
            for name in ("e2e_latency", "time_to_first_token")
        }
        value = lambda name, metrics=metrics: metrics[name]["value"]
        missing = [
            name
            for name, metric in metrics.items()
            if isinstance(metric, dict)
            and metric.get("state") in ("missing", "ambiguous")
        ]
        output.append(
            {
                "schema": "vllm_pd.request.v1",
                "request_id": request["label"],
                "session_id": None,
                "client_request_id": next(
                    (
                        row.get("client_request_id")
                        for row in request_records
                        if row.get("client_request_id") is not None
                    ),
                    None,
                ),
                "model": next(
                    (
                        row.get("model")
                        for row in request_records
                        if row.get("model") is not None
                    ),
                    None,
                ),
                "ts": wall_timestamp(request_records),
                "route": {
                    "p": "prefill" if phase == "p" else None,
                    "d": "decode" if phase == "d" else None,
                    "retries": 0,
                },
                "result": {
                    "status": api_terminal.get("status", "ok"),
                    "finish_reason": api_terminal.get("finish_reason"),
                    "http_status": api_terminal.get("http_status"),
                },
                "tokens": {
                    "prompt": value("input_tokens"),
                    "cached": value("cached_tokens"),
                    "cached_local": value("hbm_cached_tokens"),
                    "cached_ext": value("external_cached_tokens"),
                    "prefill_compute": value("prefill_compute_tokens"),
                    "output": value("output_tokens"),
                },
                "request_metrics": metrics,
                "latency_ms": {
                    "e2e": official_ms["e2e_latency"],
                    "starts_at": "engine",
                    "ttft": official_ms["time_to_first_token"],
                    "decode": value("decode_ms"),
                    "tpot": None,
                    "p_total": None,
                    "d_total": None,
                    "retry_ms": None,
                },
                "ttft": ttft,
                "tpot": {"steps": value("decode_steps"), "d": {}},
                "background": background,
                "checks": {"p_unattributed_ms": None, "d_first_unattributed_ms": None},
                "recorded_level": {
                    "p": 1 if ttft["p"] else 0,
                    "d": 1 if ttft["d"] else 0,
                },
                "attempts": 1,
                "missing": missing,
                "request_summaries": summaries,
                "associated_tasks": [task_json(task, records) for task in task_rows],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    args = parser.parse_args()
    records, errors = read_records(args.logs)
    if errors:
        raise SystemExit("invalid PD_DIAG records: " + "; ".join(errors[:3]))
    output = convert(records)
    payload: Any = output[0] if len(output) == 1 else output
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(output)} request records -> {args.out}")


if __name__ == "__main__":
    main()
