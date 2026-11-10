#!/usr/bin/env python3
"""Flatten a vLLM P/D report JSON into Grafana-friendly row arrays.

The output is a static export/API payload, not a Grafana datasource or service.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

STAGE_GROUPS = {
    "api_resolve": "api_input",
    "render": "api_input",
    "tokenize": "api_input",
    "input_process": "api_input",
    "frontend_request_registered": "api_input",
    "request_id_link": "api_input",
    "engine_submit": "api_input",
    "engine_input_received": "core_input",
    "engine_request_init": "core_input",
    "request_hash_ranges": "core_input",
    "engine_input_queue_wait": "core_input",
    "scheduler_queue_episode": "core_input",
    "scheduler_schedule": "core_input",
    "engine_execute_submit": "core_input",
    "engine_batch_enqueued": "core_input",
    "runner.input_prepare": "worker",
    "kv.load_submit": "worker",
    "runner.forward": "worker",
    "runner.logits": "worker",
    "runner.sampling": "worker",
    "runner.draft": "worker",
    "runner.output_prepare": "worker",
    "kv.save_wait": "worker",
    "engine_batch_dequeued": "core_output",
    "engine_execute_wait": "core_output",
    "engine_sample_call": "core_output",
    "engine_scheduler_update": "core_output",
    "engine_output_queue_wait": "core_output",
    "engine_output_send": "core_output",
    "api_output_received": "api_output",
    "api_output_endpoint": "api_output",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _ms(value: Any, source_unit: str = "ns") -> float | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    scale = 1_000_000 if source_unit == "ns" else 1000 if source_unit == "us" else None
    return round(number / scale, 6) if scale else None


def _observations(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for request in dashboard.get("requests", []):
        request_key = request["key"]
        side = _request_side(request)
        stages = [stage for group in request.get("mainline", [])
                  for stage in group["stages"]]
        for stage in stages:
            event = stage.get("event")
            group = STAGE_GROUPS.get(event)
            if group is None:
                continue
            for measurement in stage.get("measurements", []):
                duration = measurement.get("duration_ns")
                state = measurement.get("state", "missing")
                if state == "measured" and _ms(duration) is None:
                    state = "invalid"
                domain = measurement.get("domain")
                rows.append({"request_key": request_key, "side": side,
                             "stage_group": group,
                             "stage": stage.get("label", event), "event": event,
                             "state": state,
                             "duration_ms": (
                                 _ms(duration) if state == "measured" else None),
                             "execution_id": measurement.get("execution_id"),
                             "task_id": measurement.get("task_id"),
                             "dp_rank": _domain_rank(domain, 3),
                             "pp_rank": _domain_rank(domain, 4),
                             "tp_rank": _domain_rank(domain, 5),
                             "shared": bool(measurement.get("shared", False)),
                             "clock_domain": domain,
                             "source_file": measurement.get(
                                 "log_source", {}).get("file"),
                             "source_line": measurement.get(
                                 "log_source", {}).get("line"),
                             "note": measurement.get("note") or "; ".join(
                                 measurement.get("reasons", [])) or None})
    return rows


def _domain_rank(domain: Any, index: int) -> int | None:
    if not isinstance(domain, list) or len(domain) <= index:
        return None
    value = domain[index]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _request_side(request: dict[str, Any]) -> str | None:
    roles = set(request.get("roles", []))
    if roles & {"P", "prefill", "producer", "kv_producer"}:
        return "P"
    if roles & {"D", "decode", "consumer", "kv_consumer"}:
        return "D"
    return None


def _summary(request: dict[str, Any]) -> dict[str, Any]:
    summaries = request.get("native_summaries", [])
    return summaries[0] if len(summaries) == 1 else {}


def _summary_value(summary: dict[str, Any], *names: str) -> Any:
    for name in names:
        if summary.get(name) is not None:
            return summary[name]
    return None


def flatten(report: dict[str, Any]) -> dict[str, Any]:
    dashboard = report.get("dashboard", {})
    requests = []
    quality = []
    side_by_key = {}
    for request in dashboard.get("requests", []):
        key = request["key"]
        side = _request_side(request)
        side_by_key[key] = side
        summary = _summary(request)
        model_values = {r.get("model") for r in request.get("observations", [])
                        if r.get("model") is not None}
        model = next(iter(model_values)) if len(model_values) == 1 else None
        prompt = _summary_value(summary, "num_prompt_tokens", "prompt_tokens")
        output = _summary_value(summary, "num_generation_tokens", "output_tokens")
        ttft = summary.get("display_ms", {}).get("time_to_first_token")
        e2e = summary.get("display_ms", {}).get("e2e_latency")
        metrics = request.get("metrics", {})
        metric_values = {}
        for name, metric in metrics.items():
            if not isinstance(metric, dict) or "state" not in metric:
                continue
            metric_values[name] = metric.get("value")
            metric_values[name + "_state"] = metric.get("state")
            metric_values[name + "_unit"] = metric.get("unit")
            metric_values[name + "_source"] = metric.get("source")
        values = {"model": model, "prompt_tokens": prompt, "output_tokens": output,
                  "ttft_ms": ttft, "e2e_ms": e2e, "side": side}
        missing = [name for name, value in values.items() if value is None]
        missing.extend(name for name, metric in metrics.items()
                       if isinstance(metric, dict)
                       and metric.get("state") in ("missing", "ambiguous")
                       and name not in missing)
        status = next((item.get("status") for item in request.get("statuses", [])
                       if item.get("status") is not None), None)
        requests.append({"request_key": key, "request_id": request.get("label"),
                         "global_request_id": request.get("global_request_id"),
                         **values, **metric_values, "status": status,
                         "complete": False, "missing_fields": missing,
                         "timestamp": request.get("timestamp"),
                         "cache_source": metrics.get("cache_source"),
                         "hbm_snapshots": metrics.get("hbm_snapshots", []),
                         "hbm_physical_use": metrics.get("hbm_physical_use", []),
                         "pd_transfers": metrics.get("pd_transfers", []),
                         "metric_evidence": metrics})
        for field in missing:
            quality.append({"request_key": key, "source_file": None,
                            "source_line": None, "event": "request_field",
                            "state": "missing", "note": field})
    observations = _observations(dashboard)
    for row in observations:
        if row["state"] in ("missing", "ambiguous", "invalid"):
            quality.append({"request_key": row["request_key"],
                            "source_file": row["source_file"],
                            "source_line": row["source_line"], "event": row["event"],
                            "state": row["state"], "note": row["note"]})
    task_rows = []
    raw_tasks = {str(item.get("task_id")): item.get("observations", [])
                 for item in report.get("tasks", [])}
    hbm_rows = []
    transfer_rows = []
    for request in dashboard.get("requests", []):
        request_key = request["key"]
        metrics = request.get("metrics", {})
        for snapshot in metrics.get("hbm_snapshots", []):
            for group in snapshot.get("groups", []):
                hbm_rows.append({
                    "request_key": request_key,
                    "request_id": request.get("label"),
                    "worker_config_index": snapshot.get("worker_config_index"),
                    "group_id": group.get("group_id"),
                    "tp_rank": snapshot.get("tp_rank"),
                    "host_resident": group.get("host_resident"),
                    "referenced_unique_pages": group.get("referenced_unique_pages"),
                    "peak_referenced_unique_pages": group.get(
                        "peak_referenced_unique_pages"),
                    "request_referenced_bytes": group.get("request_referenced_bytes"),
                    "peak_request_referenced_bytes": group.get(
                        "peak_request_referenced_bytes"),
                    "physical_allocation_bytes": snapshot.get(
                        "physical_allocation_bytes"),
                    "state": "observed", "source": snapshot.get("source"),
                })
        for transfer in metrics.get("pd_transfers", []):
            transfer_rows.append({"request_key": request_key,
                                  "request_id": request.get("label"),
                                  **transfer,
                                  "duration_ms": _ms(transfer.get("duration_ns"))})
    for task in dashboard.get("tasks", []):
        task_records = raw_tasks.get(str(task.get("task_id")), [])
        for key in task.get("request_keys", []):
            for stage in task.get("stages", []):
                duration = stage.get("duration_ns")
                state = stage.get("state", "missing")
                task_rows.append({"request_key": key, "side": side_by_key.get(key),
                                  "task_id": task.get("task_id"),
                                  "operation": ",".join(task.get("operation", [])),
                                  "status": ",".join(task.get("statuses", [])),
                                  "tp_rank": _task_rank(task_records),
                                  "stage": stage.get("label", stage.get("event")),
                                  "state": state,
                                  "duration_ms": (
                                      _ms(duration) if state == "measured" else None),
                                  "keys": None, "bytes": None, "native_phase": None})
    return {
        "schema": "vllm_pd.grafana.v1", "complete": False,
        "requests": requests, "observations": observations, "tasks": task_rows,
        "hbm": hbm_rows, "pd_transfers": transfer_rows,
        "quality": quality,
        "report_quality": {"parse_errors": report.get("parse_errors", []),
                           "unassociated_count": len(report.get("unassociated", [])),
                           "unresolved_links": dashboard.get("unresolved_links", []),
                           "unresolved_native_links": dashboard.get(
                               "native_links", [])},
    }


def _task_rank(records: list[dict[str, Any]]) -> int | None:
    values = {record.get("tp_rank") for record in records
              if record.get("tp_rank") is not None}
    return next(iter(values)) if len(values) == 1 else None



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    payload = flatten(report)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{len(payload['requests'])} requests; "
          f"{len(payload['observations'])} observations -> {args.out}")


if __name__ == "__main__":
    main()
