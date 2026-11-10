#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export P/D observations without adding parallel ranks or shared batches.

Usage: python tools/pd_metrics_report.py service.log -o report.json
Multiple log files may be supplied. --html renders the same grouped evidence.
"""

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "[PD_DIAG] "
_GLOG_THREAD = re.compile(
    r"\b[IWEF]\d{4}(?:\d{4})?\s+\d{2}:\d{2}:\d{2}\.\d+"
    r"\s+(\d+)\s+\S+:\d+\]"
)
_GLOG_WALL = re.compile(
    r"\b[IWEF](\d{8}|\d{4})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{1,9})\s+"
)


def clock(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(k) for k in ("host", "pid", "run_id"))


def _native_timing(line: str) -> dict[str, Any] | None:
    marker = next((m for m in ("[BatchPutTiming]", "[BatchUpsertTiming]")
                   if m in line), None)
    if marker is None:
        return None
    prefix, body = line.split(marker, 1)
    fields = dict(re.findall(r"(\w+)=([^\s]+)", body))
    thread = re.search(r"\b(?:native_tid|native_thread_id|tid)=(\d+)", prefix)
    thread = thread or _GLOG_THREAD.search(prefix)
    process = re.search(r"\bpid=(\d+)", prefix)
    wall = _GLOG_WALL.search(prefix)
    durations = {}
    for name in (
        "phase1_start_batch_put_us", "phase2_submit_transfers_us",
        "phase3_wait_transfers_us", "phase4_finalize_batch_put_us", "total_us",
    ):
        if name in fields:
            value = float(fields[name])
            if not _number(value):
                raise ValueError(f"invalid native duration: {name}")
            durations[name.removesuffix("_us")] = value * 1000
    if not durations:
        raise ValueError("native timing record has no durations")
    return {
        "version": 1, "event": "mooncake_native_timing",
        "stage_path": "native.mooncake.put", "duration_fields_ns": durations,
        "timing_scope": fields.get("timing_scope", "unspecified"),
        "native_kind": marker, "native_fields": fields,
        "native_tid": int(thread[1]) if thread else None,
        "pid": int(process[1]) if process else None,
        "glog_wall": list(wall.groups()) if wall else None,
    }


def read_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                if MARKER not in line and not any(
                    m in line for m in ("[BatchPutTiming]", "[BatchUpsertTiming]")
                ):
                    continue
                try:
                    record = (json.loads(line.split(MARKER, 1)[1])
                              if MARKER in line else _native_timing(line))
                    if not isinstance(record, dict) or record.get("version") != 1:
                        raise ValueError("unsupported diagnostic record")
                    if not isinstance(record.get("event"), str):
                        raise ValueError("missing event")
                except (ValueError, TypeError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
                    continue
                # Only explicit record IDs prove duplicate emission. Identical
                # counters without an ID may be different real observations.
                identity = record.get("record_id")
                if identity is not None:
                    key = (*clock(record), str(identity))
                    if key in seen:
                        continue
                    seen.add(key)
                record["log_source"] = {"file": str(path), "line": line_no}
                records.append(record)
    return records, errors


def union_duration(records: list[dict[str, Any]]) -> int | None:
    """Union only validated ranges from one explicitly identified clock."""
    if not records:
        return None
    if any(r.get("event") in _POSITION_EVENTS for r in records):
        return None
    domains = {clock(r) for r in records}
    if len(domains) != 1 or any(v is None for v in next(iter(domains))):
        return None
    intervals = []
    for record in records:
        a, b = record.get("start_ns"), record.get("end_ns")
        if (
            isinstance(a, bool)
            or isinstance(b, bool)
            or not isinstance(a, int)
            or not isinstance(b, int)
            or b < a
        ):
            return None
        intervals.append((a, b))
    total = 0
    right = None
    for a, b in sorted(intervals):
        total += max(0, b - max(a, right if right is not None else a))
        right = max(right, b) if right is not None else b
    return total


def summarize(records: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    executions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unassociated = []
    for record in records:
        execution = record.get("execution_id")
        task = record.get("task_id")
        request = record.get("observation_id") or record.get("api_request_id")
        if request is None and record.get("request_id") is not None:
            request = json.dumps(
                [*clock(record), record["request_id"]], separators=(",", ":")
            )
        associated = False
        for identity, groups in (
            (execution, executions),
            (task, tasks),
            (request, requests),
        ):
            if identity is not None:
                groups[str(identity)].append(record)
                associated = True
        if not associated:
            unassociated.append(record)
    execution_rows = []
    for identity, group in executions.items():
        ranks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        shared = []
        members = set()
        for record in group:
            ids = record.get("request_ids")
            if isinstance(ids, (list, tuple)):
                members.update(str(r) for r in ids)
            if record.get("request_id") is not None:
                members.add(str(record["request_id"]))
            rank = record.get("tp_rank")
            if rank is None:
                shared.append(record)
            else:
                key = json.dumps([*clock(record), rank], separators=(",", ":"))
                ranks[key].append(record)
        rank_rows = []
        for domain, observations in ranks.items():
            events: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in observations:
                events[record["event"]].append(record)
            rank_rows.append(
                {
                    "domain": json.loads(domain),
                    "event_union_ns": {
                        event: union_duration(items) for event, items in events.items()
                    },
                    "observations": observations,
                }
            )
        execution_rows.append(
            {
                "execution_id": identity,
                "request_ids": sorted(members),
                "shared_observations": shared,
                "ranks": rank_rows,
            }
        )
    report = {
        "schema_version": 1,
        "complete": False,
        "completeness_note": (
            "Observed evidence only; log presence does not prove complete capture. "
            "Shared batch ranges are not exclusive request latency; parallel TP "
            "ranges and cross-process clocks are never added. Bytes retain their "
            "producer semantics and are not inferred wire throughput."
        ),
        "parse_errors": errors,
        "event_counts": dict(Counter(r["event"] for r in records)),
        "native_timings": [r for r in records if r["event"] == "mooncake_native_timing"],
        "identity_links": [
            r
            for r in records
            if r["event"]
            in (
                "api_request_link",
                "request_id_link",
                "request_child_link",
                "pd_request_link",
                "pd_transfer_offer",
                "pool_task_member",
            )
        ],
        "requests": [{"identity": k, "observations": v} for k, v in requests.items()],
        "executions": execution_rows,
        "tasks": [{"task_id": k, "observations": v} for k, v in tasks.items()],
        "unassociated": unassociated,
    }
    report["dashboard"] = build_dashboard(records)
    return report


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


_POSITION_EVENTS = {
    "engine_input_received", "engine_input_enqueued", "engine_input_dequeued",
    "engine_output_enqueued", "engine_output_dequeued",
    "engine_batch_enqueued", "engine_batch_dequeued",
    "api_output_received", "api_output_endpoint",
    "frontend_request_registered", "request_id_link",
}


def _measurement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One clock/event only; never turn partial evidence into a complete total."""
    result: dict[str, Any] = {"state": "missing", "duration_ns": None,
                              "observations": len(records)}
    if not records:
        return result
    if all(r.get("event") in _POSITION_EVENTS for r in records):
        result.update(state="observed", note="瞬时位置证据，不是计时器")
        return result
    statuses = [r.get("status") for r in records]
    if all(s == "skipped" for s in statuses):
        result.update(state="not_applicable",
                      reasons=list(dict.fromkeys(r.get("reason") for r in records)))
        return result
    if all(s == "fused" for s in statuses):
        result.update(state="fused",
                      reasons=list(dict.fromkeys(r.get("reason") for r in records)))
        return result
    if any(r.get("valid") is False or r.get("complete") is False for r in records):
        result["note"] = "区间无效或采集不完整"
        return result
    # Ranges, when present, take precedence over a possibly stale duration.
    if any("start_ns" in r or "end_ns" in r for r in records):
        value = union_duration(records)
    elif len(records) == 1 and _number(records[0].get("duration_ns")):
        value = records[0]["duration_ns"]
    else:
        value = None
    if value is not None:
        result.update(state="measured", duration_ns=value)
    else:
        result["note"] = "缺少有效同进程区间；未补零、未累加计数或跨进程时间"
    return result


def _stage_rows(records: list[dict[str, Any]], events: dict[str, str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record["event"] in events:
            category = record.get("status")
            category = category if category in ("skipped", "fused") else "timed"
            groups[(*clock(record), record["event"], category)].append(record)
    return [
        {"event": key[-2], "label": events[key[-2]], "domain": list(key[:-2]),
         "shared": key[-2].startswith(
             ("engine_output_", "engine_batch_", "api_output_"))
             or key[-2] == "engine_scheduler_update",
         **_measurement(group)}
        for key, group in groups.items()
    ]

def _queue_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair object residence, never the time spent blocking on an empty get."""
    rows = []
    for name, fallback in (("input", "request_id"), ("output", "output_batch_id")):
        events = (f"engine_{name}_enqueued", f"engine_{name}_dequeued")
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for index, record in enumerate(records):
            if record["event"] not in events:
                continue
            field = ("input_message_id" if name == "input"
                     and record.get("input_message_id") is not None else fallback)
            identity = record.get(field)
            # An unknown clock or object ID must not pair with another unknown.
            key = (*clock(record), field, identity)
            if identity is None or any(v is None for v in clock(record)):
                key = (*key, index)
            groups[key].append(record)
        for group in groups.values():
            enqueued = [r for r in group if r["event"] == events[0]]
            dequeued = [r for r in group if r["event"] == events[1]]
            first = group[0]
            field = ("input_message_id" if name == "input"
                     and first.get("input_message_id") is not None else fallback)
            metric = {"state": "missing", "duration_ns": None,
                      "note": "缺少唯一、同域、同对象的有效入队/出队位置"}
            if (len(enqueued) == len(dequeued) == 1
                    and len({clock(r) for r in group}) == 1
                    and all(v is not None for v in clock(first))
                    and first.get(field) is not None
                    and (name == "output" or (
                        first.get("request_id") is not None
                        and len({r.get("request_id") for r in group}) == 1))):
                start, end = (enqueued[0].get("queue_enter_ns"),
                              dequeued[0].get("queue_exit_ns"))
                if (_number(start) and _number(end) and end >= start
                        and all(r.get("valid") is not False
                                and r.get("complete") is not False for r in group)):
                    metric = {"state": "measured", "duration_ns": end - start,
                              "note": "对象在本进程队列的驻留；不含空队列 get 阻塞"}
            rows.append({
                "event": f"engine_{name}_queue_wait", "domain": list(clock(first)),
                field: first.get(field), "request_id": first.get("request_id"),
                "shared": name == "output",
                "request_ids": list(dict.fromkeys(
                    value for r in group for value in r.get("request_ids", []))),
                "evidence": group, **metric,
            })
    return rows



def _worker_stages(records: list[dict], labels: dict[str, str],
                   domain: tuple) -> list[dict]:
    """Nest only intervals proven to enclose each other on the same thread."""
    rows = _stage_rows(records, labels)
    for row in rows:
        row["domain"] = list(domain)
        row["parent_event"] = None
        children = [r for r in records if r["event"] == row["event"]]
        enclosing = []
        for other in rows:
            if other is row:
                continue
            parents = [r for r in records if r["event"] == other["event"]]
            widths = []
            for child in children:
                candidates = [
                    p["end_ns"] - p["start_ns"] for p in parents
                    if all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool)
                           for r in (p, child) for k in ("start_ns", "end_ns"))
                    and child.get("native_tid") is not None
                    and child.get("native_tid") == p.get("native_tid")
                    and p["start_ns"] <= child["start_ns"] <= child["end_ns"] <= p["end_ns"]
                    and p["end_ns"] - p["start_ns"] > child["end_ns"] - child["start_ns"]
                ]
                if not candidates:
                    break
                widths.append(min(candidates))
            else:
                if widths:
                    enclosing.append((max(widths), other["event"]))
        if enclosing:
            row["parent_event"] = min(enclosing)[1]
    output = []

    def append(parent: str | None, depth: int) -> None:
        for row in rows:
            if row["parent_event"] == parent:
                row["depth"] = depth
                output.append(row)
                append(row["event"], depth + 1)

    append(None, 0)
    return output



def _worker_coverage(records: list[dict]) -> dict[str, dict]:
    """Compare each Worker method with observed same-thread stage intervals."""
    stages = {
        "runner.input_prepare",
        "runner.forward",
        "runner.logits",
        "runner.sampling",
        "runner.draft",
        "runner.output_prepare",
        "runner.postprocess",
        "kv.load_submit",
        "kv.save_wait",
        "kv.finish_forward",
        "kv.transfer_results",
        "kv.mamba_state_copy",
    }
    result = {}
    for name, event in (
        ("execute", "worker.execute"),
        ("sample", "worker.sample_tokens"),
    ):
        outer = [record for record in records if record["event"] == event]
        metric = {
            "duration_ns": _measurement(outer).get("duration_ns"),
            "covered_ns": None,
            "uncovered_ns": None,
            "state": "missing",
        }
        result[name] = metric
        if (
            len(outer) != 1
            or metric["duration_ns"] is None
            or outer[0].get("native_tid") is None
            or outer[0].get("status") != "ok"
        ):
            continue
        parent = outer[0]
        children = []
        invalid = False
        for child in records:
            if (
                child["event"] not in stages
                or clock(child) != clock(parent)
                or child.get("native_tid") != parent["native_tid"]
            ):
                continue
            start, end = child.get("start_ns"), child.get("end_ns")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or end < start
            ):
                invalid = True
                continue
            if end <= parent["start_ns"] or start >= parent["end_ns"]:
                continue
            if (
                start < parent["start_ns"]
                or end > parent["end_ns"]
                or child.get("status") != "ok"
                or child.get("complete") is False
            ):
                invalid = True
            else:
                children.append(child)
        covered = union_duration(children) if children and not invalid else None
        if covered is not None:
            metric.update(
                covered_ns=covered,
                uncovered_ns=metric["duration_ns"] - covered,
                state="measured",
            )
    return result


def _native_inside_call(native: dict, call: dict) -> bool:
    wall = native.get("glog_wall")
    start, end = call.get("start_wall_ns"), call.get("end_wall_ns")
    offset = call.get("wall_utc_offset_seconds")
    if (
        not wall
        or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in (start, end, offset)
        )
        or end < start
        or abs(offset) >= 86400
    ):
        return False
    for native_field, call_field in (
        ("keys", "submitted_keys"),
        ("bytes", "submitted_bytes"),
    ):
        value = native.get("native_fields", {}).get(native_field)
        if value is not None and (
            not value.isdecimal() or int(value) != call.get(call_field)
        ):
            return False
    date, hour, minute, second, fraction = wall
    if len(date) == 8:
        years = [int(date[:4])]
        date = date[4:]
    else:
        years = {
            datetime.fromtimestamp(value // 1_000_000_000 + offset, timezone.utc).year
            for value in (start, end)
        }
    for year in years:
        try:
            stamp = datetime(
                year,
                int(date[:2]),
                int(date[2:]),
                int(hour),
                int(minute),
                int(second),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        instant = (int(stamp.timestamp()) - offset) * 1_000_000_000
        instant += int(fraction.ljust(9, "0"))
        precision = 10 ** (9 - len(fraction))
        if start <= instant + precision - 1 and instant <= end:
            return True
    return False


def _native_thread_links(records: list[dict], tasks: list[dict]) -> list[dict]:
    """Resolve native calls by thread, wall boundaries, and submission counts."""
    threads: dict[tuple, dict[tuple, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_by_record = {
        index: task["key"] for task in tasks for index in task["record_indices"]
    }
    for index, record in enumerate(records):
        source = record.get("log_source", {}).get("file")
        tid = record.get("native_tid")
        if (
            record["event"] == "mooncake_native_timing"
            or source is None
            or tid is None
            or any(value is None for value in clock(record))
        ):
            continue
        threads[(source, tid)][clock(record)].append(index)
    links = []
    for record in records:
        if record["event"] != "mooncake_native_timing":
            continue
        source = record.get("log_source", {}).get("file")
        candidates = []
        for identity, indices in threads.get(
            (source, record.get("native_tid")), {}
        ).items():
            if record.get("pid") is not None and identity[1] != record["pid"]:
                continue
            owner = {
                "domain": list(identity),
                "native_tid": record["native_tid"],
                "evidence_record_indices": indices,
            }
            for field in ("tp_rank", "dp_rank", "pp_rank"):
                ranks = {
                    records[i][field]
                    for i in indices
                    if records[i].get(field) is not None
                }
                owner[field] = next(iter(ranks)) if len(ranks) == 1 else None
            task_indices = [i for i in indices if i in task_by_record]
            task_keys = list(dict.fromkeys(task_by_record[i] for i in task_indices))
            owner["task_candidates"] = task_keys
            owner["task_key"] = task_keys[0] if len(task_keys) == 1 else None
            owner["task_evidence_record_indices"] = task_indices
            candidates.append(owner)
        matching_calls = [
            index
            for owner in candidates
            for index in owner["evidence_record_indices"]
            if records[index]["event"] == "pool_native_call"
            and _native_inside_call(record, records[index])
        ]
        call_index = matching_calls[0] if len(matching_calls) == 1 else None
        owner = candidates[0] if len(candidates) == 1 else None
        if owner is None and call_index is not None:
            owner = next(
                candidate for candidate in candidates
                if candidate["domain"] == list(clock(records[call_index]))
            )
        if owner is None:
            task_owners = [candidate for candidate in candidates if candidate["task_key"]]
            if len(task_owners) == 1:
                owner = task_owners[0]
        task_owner = owner.get("task_key") if owner is not None else None
        links.append(
            {
                "native": record,
                "association": (
                    "same_thread_wall_enclosure"
                    if call_index is not None
                    else "same_file_unique_thread_owner"
                    if len(candidates) == 1
                    else "ambiguous_thread_owner"
                    if candidates
                    else "missing_thread_owner"
                ),
                "thread_owner": owner,
                "thread_candidates": candidates,
                "call_association": (
                    "same_thread_wall_enclosure"
                    if call_index is not None
                    else "ambiguous_call_boundary"
                    if matching_calls
                    else "unavailable_no_call_boundary"
                ),
                "call_record_index": call_index,
                "task_key": task_by_record.get(call_index) if call_index is not None else None,
                "task_association": (
                    "unique_call_boundary"
                    if call_index is not None
                    else "unique_thread_task"
                    if task_owner is not None
                    else "unresolved"
                ),
                "task_candidate_keys": (
                    owner.get("task_candidates", []) if owner is not None else []
                ),
            }
        )
    return links


def _request_metrics(request: dict, observations: list[dict]) -> dict:
    """Expose request facts without merging clocks, ranks or transfer scopes."""
    metrics = {}

    def scalar(name, values, unit, source, note=None):
        valid = [value for value in values if _number(value)]
        unique = set(valid)
        state = "measured" if len(unique) == 1 else "ambiguous" if unique else "missing"
        metrics[name] = {
            "value": valid[0] if state == "measured" else None,
            "unit": unit, "state": state, "source": source, "note": note,
        }

    summaries = [r for r in observations if r["event"] == "request_summary"]
    scalar("input_tokens", [r.get("num_prompt_tokens") for r in summaries],
           "tokens", "request_summary.num_prompt_tokens")
    scalar("output_tokens", [r.get("num_generation_tokens") for r in summaries],
           "tokens", "request_summary.num_generation_tokens")
    scalar("cached_tokens", [r.get("num_cached_tokens") for r in summaries],
           "tokens", "request_summary.num_cached_tokens")
    scalar("decode_ms", [r.get("decode_time") * 1000 for r in summaries
                         if _number(r.get("decode_time"))],
           "ms", "request_summary.decode_time",
           "官方首 token 到末 token 的 wall 时间；不是 forward 累计")
    lookups = [r for r in observations if r["event"] == "request_cache_lookup"]
    for name, field in (("hbm_cached_tokens", "num_local_cached_tokens"),
                        ("external_cached_tokens", "num_external_cached_tokens"),
                        ("prefill_compute_tokens", "num_prompt_uncached_tokens")):
        scalar(name, [r.get(field) for r in lookups], "tokens",
               "request_cache_lookup." + field,
               "逐次 lookup 的有效命中边界，不证明 load 成功或发生了 forward")
    decode_steps = set()
    decode_request_ids = {alias["value"] for alias in request["aliases"]
                          if alias["kind"] == "request_id"}
    for record in observations:
        if record["event"] != "scheduler_execution":
            continue
        for member in record.get("members", []):
            before = member.get("num_computed_tokens_before")
            prompt = member.get("num_prompt_tokens")
            scheduled = member.get("scheduled_tokens")
            if (member.get("request_id") in decode_request_ids
                    and _number(before) and _number(prompt) and _number(scheduled)
                    and before >= prompt and scheduled > 0
                    and record.get("execution_id") is not None):
                decode_steps.add((*clock(record), record["execution_id"]))
    metrics["observed_decode_steps"] = {
        "value": len(decode_steps), "unit": "steps", "state": "observed",
        "source": "scheduler_execution.members",
        "note": "输入日志中的 decode 调度次数；MTP 多 token 仍记一步",
    }
    terminal = [r for r in observations if r["event"] == "scheduler_queue_summary"]
    scalar("decode_steps", [r.get("decode_steps") for r in terminal],
           "steps", "scheduler_queue_summary.decode_steps",
           "整个请求的 decode 调度次数，不含 prefill；不等同成功完成的 forward 次数")
    metrics["summaries"] = summaries
    metrics["cache_lookups"] = lookups
    metrics["hbm_snapshots"] = [
        r for r in observations if r["event"] in ("kv_cache_request", "request_kv_hbm")
    ]
    metrics["hbm_physical_use"] = [
        r for r in observations if r["event"] == "kv_cache_physical_use"
    ]
    transfers = [r for r in observations
                 if r["event"] in ("pd_transfer_batch_finished",
                                    "pd_transfer_request_finished")]
    metrics["pd_transfers"] = []
    for record in transfers:
        batch = record["event"] == "pd_transfer_batch_finished"
        if batch:
            duration = _measurement([record]).get("duration_ns")
        else:
            value = record.get("transfer_duration_ns")
            duration = value if _number(value) else None
        metrics["pd_transfers"].append({
            "event": record["event"],
            "request_generation": record.get("request_generation"),
            "task_id": record.get("task_id"), "domain": list(clock(record)),
            "tp_rank": record.get("tp_rank"), "dp_rank": record.get("dp_rank"),
            "pp_rank": record.get("pp_rank"), "shared": False,
            "status": record.get("status"),
            "submitted_bytes": record.get("submitted_bytes"),
            "succeeded_bytes": record.get("succeeded_bytes"),
            "failed_bytes": record.get("failed_bytes"), "duration_ns": duration,
            "scope": "native_batch" if batch else "request_generation_envelope",
            "log_source": record.get("log_source"),
            "byte_semantics": record.get("byte_semantics") or
                              "submitted_lengths_classified_by_batch_status",
        })
    request_transfers = [r for r in metrics["pd_transfers"]
                         if r["event"] == "pd_transfer_request_finished"]
    scalar("pd_transfer_ms", [r["duration_ns"] / 1e6 for r in request_transfers
                              if r["duration_ns"] is not None],
           "ms", "pd_transfer_request_finished.transfer_duration_ns",
           "首末 native batch 时间包络，含批间空档；不同 TP/generation 不合并")
    if len(request_transfers) > 1:
        metrics["pd_transfer_ms"].update(value=None, state="ambiguous")
    metrics["cache_source"] = {
        "hbm": metrics["hbm_cached_tokens"],
        "connector": metrics["external_cached_tokens"],
        "mooncake_local_tokens": None, "mooncake_remote_tokens": None,
        "note": "connector 外部命中；Mooncake exists 不报告本地/远端位置",
    }
    metrics["note"] = "HBM 为块引用容量快照；PD 为提交载荷；TP 行与重试不合并"
    return metrics


def build_dashboard(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derived display data; raw v1 evidence and its grouping stay unchanged.

    Request aliases are local to an identified process/API observation.
    Cross-process links require explicit observations, execution membership,
    unique randomized EngineCoreRequest submission/receipt, or remote offers.
    Lists retain first encounter order, never cross-host timestamp order.
    """
    parent: dict[tuple, tuple] = {}
    node_order: dict[tuple, int] = {}
    record_nodes: list[list[tuple]] = []

    def domain(record: dict, index: int) -> tuple:
        values = clock(record)
        # Incomplete process identities cannot safely join even within a file.
        return values if all(v is not None for v in values) else (
            "unidentified", index, None)

    def node(kind: str, value: Any, record: dict, index: int) -> tuple | None:
        if value is None:
            return None
        local = domain(record, index)
        if record.get("observation_id") is not None:
            local = (*local, str(record["observation_id"]))
        key = (kind, str(value)) if kind == "observation_id" else (
            kind, local, str(value))
        if key not in parent:
            parent[key] = key
            node_order[key] = len(node_order)
        return key

    def root(key: tuple) -> tuple:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(nodes: list[tuple]) -> None:
        if nodes:
            roots = [root(n) for n in nodes]
            first = min(roots, key=node_order.__getitem__)
            for other in roots:
                parent[other] = first

    # Request aliases are local to identified API/process observations.
    for index, record in enumerate(records):
        nodes = []
        for kind in ("observation_id", "api_request_id", "request_id",
                     "external_req_id"):
            value = record.get(kind)
            if value is not None:
                nodes.append(node(kind, value, record, index))
        join(nodes)
        if record["event"] == "api_request_link":
            api_key = node("api_request_id", record.get("api_request_id"),
                           record, index)
            external_key = node("external_req_id", record.get("external_req_id"),
                                 record, index)
            join([key for key in (api_key, external_key) if key is not None])
        else:
            join(nodes)
        if record["event"] == "request_child_link":
            ancestor = node("request_id", record.get("parent_request_id"),
                            record, index)
            join(nodes + ([ancestor] if ancestor is not None else []))
        shared_api_batch = (
            record["event"] in ("api_output_received", "api_output_endpoint")
            and record.get("batch_shared") is True
        )
        if not shared_api_batch:
            for value in record.get("request_ids", []):
                node("request_id", value, record, index)
        record_nodes.append(nodes)
    api_roots_by_external: dict[str, set[tuple]] = defaultdict(set)
    core_roots_by_external: dict[str, set[tuple]] = defaultdict(set)
    for index, record in enumerate(records):
        external = record.get("external_req_id")
        if external is None:
            continue
        if record["event"] == "api_request_link":
            api_roots_by_external[str(external)].update(
                root(alias) for alias in record_nodes[index]
                if alias[0] == "api_request_id"
            )
        elif record["event"] == "request_id_link":
            core_roots_by_external[str(external)].update(
                root(alias) for alias in record_nodes[index]
                if alias[0] == "request_id"
                and alias[-1] == str(record.get("request_id"))
            )
    for external in api_roots_by_external.keys() & core_roots_by_external.keys():
        api_roots = api_roots_by_external[external]
        core_roots = core_roots_by_external[external]
        if len(api_roots) == len(core_roots) == 1:
            join([*api_roots, *core_roots])

    executions: dict[str, dict] = {}
    execution_anchors: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        identity = record.get("execution_id")
        if identity is None:
            continue
        identity = str(identity)
        execution = executions.setdefault(identity, {
            "execution_id": identity, "record_indices": [], "request_keys": [],
            "members": [], "shared": [], "workers": [], "no_forward": None,
        })
        execution["record_indices"].append(index)
        if record["event"] in ("scheduler_execution", "scheduler_schedule",
                               "engine_execution"):
            execution_anchors[identity].append((record, index))
            ids = record.get("request_ids", [])
            if isinstance(ids, list):
                for value in ids:
                    node("request_id", value, record, index)
        for member in record.get("members", []):
            if isinstance(member, dict) and member.get("request_id") is not None:
                node("request_id", member["request_id"], record, index)

    # Without Scheduler logs, explicit Worker execution membership is still
    # usable. Choose one observed process as the alias anchor, not a timestamp.
    for identity, execution in executions.items():
        if execution_anchors.get(identity):
            continue
        for index in execution["record_indices"]:
            record = records[index]
            if record.get("request_ids") and all(v is not None for v in clock(record)):
                execution_anchors[identity].append((record, index))
                for value in record["request_ids"]:
                    node("request_id", value, record, index)
                execution["anchor_source"] = "worker_execution_membership_only"
                break
    completion_ids = {}
    for identity, execution in executions.items():
        completed = list(dict.fromkeys(
            value for index in execution["record_indices"]
            for field in ("finished_sending", "finished_recving", "failed_recving")
            for value in records[index].get(field, [])
        ))
        execution["completion_request_ids"] = completed
        completion_ids[identity] = completed
        anchors = execution_anchors.get(identity, [])
        if len({domain(r, i) for r, i in anchors}) == 1:
            for value in completed:
                node("request_id", value, anchors[0][0], anchors[0][1])

    def execution_nodes(identity: Any, request_id: Any) -> list[tuple]:
        if identity is None or request_id is None:
            return []
        anchors = execution_anchors.get(str(identity), [])
        domains = {domain(r, i) for r, i in anchors}
        # A reused execution ID in independent scheduler processes is ambiguous.
        if len(domains) != 1:
            return []
        candidates = []
        for record, index in anchors:
            if (request_id in record.get("request_ids", [])
                    or request_id in completion_ids.get(str(identity), [])):
                candidate = node("request_id", request_id, record, index)
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    # Explicit execution membership bridges Scheduler -> Worker / deferred task.
    # Do not join multiple members of one mixed batch to each other.
    for index, record in enumerate(records):
        if record.get("request_id") is not None:
            candidates = execution_nodes(record.get("execution_id"),
                                         record["request_id"])
            join(record_nodes[index] + candidates)
        if record["event"] not in ("api_output_received", "api_output_endpoint"):
            for value in record.get("request_ids", []):
                local = node("request_id", value, record, index)
                join([local] + execution_nodes(record.get("execution_id"), value))

    # assign_request_id emits the randomized transport identity before
    # AsyncLLM sends that exact EngineCoreRequest to EngineCore.add_request.
    # Require the emitted origin, not just a UUID-looking bare request string.
    origins: set[tuple] = set()
    submits: dict[str, list[int]] = defaultdict(list)
    receives: dict[str, list[int]] = defaultdict(list)
    transport_links = []
    transport_candidates = []
    for index, record in enumerate(records):
        internal = record.get("request_id")
        if internal is None:
            continue
        internal = str(internal)
        if record["event"] == "request_id_link":
            external = record.get("external_req_id")
            suffix = internal.removeprefix(str(external) + "-")
            if (external is not None and internal != external and len(suffix) == 8
                    and all(c in "0123456789abcdef" for c in suffix)
                    and record_nodes[index]):
                origins.add(root(record_nodes[index][0]))
        elif record["event"] == "engine_submit":
            submits[internal].append(index)
        elif record["event"] == "engine_core_received":
            receives[internal].append(index)
    for internal, sender_indices in submits.items():
        receiver_indices = receives.get(internal, [])
        sender_roots = {root(record_nodes[i][0]) for i in sender_indices
                        if record_nodes[i]}
        sender_domains = {clock(records[i]) for i in sender_indices}
        receiver_domains = {clock(records[i]) for i in receiver_indices}
        complete_domains = all(
            all(value is not None for value in clock(records[i]))
            for i in sender_indices + receiver_indices)
        proven = (
            len(sender_roots) == 1
            and next(iter(sender_roots)) in {root(n) for n in origins}
            and len(sender_indices) == 1 and len(receiver_indices) == 1
            and len(sender_domains) == 1 and len(receiver_domains) == 1
            and sender_domains != receiver_domains and complete_domains
        )
        if proven:
            sender, receiver = sender_indices[0], receiver_indices[0]
            join(record_nodes[sender] + record_nodes[receiver])
            transport_links.append({
                "request_id": internal, "sender_record_index": sender,
                "receiver_record_index": receiver,
                "source": "randomized_request_id_link+engine_submit+engine_core_received",
            })
        elif receiver_indices:
            transport_candidates.append({
                "request_id": internal, "sender_record_indices": sender_indices,
                "receiver_record_indices": receiver_indices,
                "reason": "随机化来源、唯一提交/接收或完整进程身份不足；仅候选，不归入耗时",
            })

    # API output batches carry EngineCore request IDs. Attach them only after
    # the explicit request-id, submit, and receive bridge has joined the API
    # request's randomized transport identity. Never infer a link from clocks.
    for index, record in enumerate(records):
        if record["event"] not in ("api_output_received", "api_output_endpoint"):
            continue
        member_ids = record.get("request_ids", [])
        linked = set()
        for value in member_ids:
            matches = [link for link in transport_links
                       if link["request_id"] == str(value)]
            if len(matches) != 1:
                linked = set()
                break
            sender = matches[0]["sender_record_index"]
            linked.add(root(record_nodes[sender][0]))
        if linked:
            record_nodes[index].extend(
                alias for alias in parent if root(alias) in linked
            )
    offers: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["event"] == "pd_transfer_offer":
            engine = record.get("transfer_engine_id", record.get("engine_id"))
            request = record.get("transfer_request_id", record.get("request_id"))
            if engine is not None and request is not None:
                offers[(str(engine), str(request))].extend(record_nodes[index])
    unresolved_links = []
    for index, record in enumerate(records):
        if record["event"] != "pd_request_link":
            continue
        key = (str(record.get("remote_engine_id")),
               str(record.get("remote_request_id")))
        targets = offers.get(key, [])
        target_roots = {root(n) for n in targets}
        if len(target_roots) == 1:
            join(record_nodes[index] + targets)
        else:
            unresolved_links.append({
                "record_index": index, "reason": "远端 offer 缺失或身份歧义",
                "remote_engine_id": record.get("remote_engine_id"),
                "remote_request_id": record.get("remote_request_id"),
            })

    requests: dict[tuple, dict] = {}
    for alias in parent:
        identity = root(alias)
        request = requests.setdefault(identity, {
            "key": f"request-{len(requests) + 1}", "aliases": [],
            "record_indices": [], "execution_ids": [], "task_keys": [],
        })
        request["aliases"].append({
            "kind": alias[0], "value": alias[-1],
            "domain": None if alias[0] == "observation_id" else list(alias[1]),
        })
    for index, nodes in enumerate(record_nodes):
        if nodes:
            requests[root(nodes[0])]["record_indices"].append(index)
    for candidate in transport_candidates:
        candidate["sender_keys"] = list(dict.fromkeys(
            requests[root(record_nodes[i][0])]["key"]
            for i in candidate["sender_record_indices"] if record_nodes[i]))
        candidate["receiver_keys"] = list(dict.fromkeys(
            requests[root(record_nodes[i][0])]["key"]
            for i in candidate["receiver_record_indices"] if record_nodes[i]))
        requests_by_key = {r["key"]: r for r in requests.values()}
        for key in candidate["receiver_keys"]:
            requests_by_key[key].setdefault("candidate_for", []).extend(
                candidate["sender_keys"])

    def keys_for(record: dict, index: int) -> list[str]:
        nodes = list(record_nodes[index])
        ids = ([] if record["event"] in ("api_output_received", "api_output_endpoint")
               and record.get("batch_shared") is True
               else list(record.get("request_ids", [])))
        for field in ("finished_sending", "finished_recving", "failed_recving"):
            ids.extend(record.get(field, []))
        for value in ids:
            local = ("request_id",
                     (*domain(record, index), str(record["observation_id"]))
                     if record.get("observation_id") is not None
                     else domain(record, index), str(value))
            if local in parent:
                nodes.append(local)
            nodes.extend(execution_nodes(record.get("execution_id"), value))
        for member in record.get("members", []):
            if isinstance(member, dict):
                nodes.extend(execution_nodes(record.get("execution_id"),
                                             member.get("request_id")))
        return list(dict.fromkeys(requests[root(n)]["key"] for n in nodes))

    # Shared batches contribute evidence to each member without joining members.
    requests_by_key = {request["key"]: request for request in requests.values()}
    for index, record in enumerate(records):
        for key in keys_for(record, index):
            request = requests_by_key[key]
            if index not in request["record_indices"]:
                request["record_indices"].append(index)

    output_batches: dict[str, dict] = {}
    for index, record in enumerate(records):
        identity = record.get("output_batch_id")
        if identity is None or record["event"] not in {
            "engine_output_enqueued", "engine_output_dequeued", "engine_output_send",
            "api_output_received", "api_output_endpoint",
        }:
            continue
        batch = output_batches.setdefault(str(identity), {
            "output_batch_id": identity, "shared": True, "request_keys": [],
            "evidence": [], "duration_ns": None,
            "note": "批次身份关联证据；不将 Core send 与 API receive 跨进程相减",
        })
        batch["evidence"].append({
            "record_index": index, "event": record["event"],
            "domain": list(clock(record)), "timestamp_ns": record.get("timestamp_ns"),
            "iteration_timestamp": record.get("iteration_timestamp"),
            "received_ns": record.get("received_ns"),
            "queue_enter_ns": record.get("queue_enter_ns"),
            "queue_exit_ns": record.get("queue_exit_ns"),
            "request_ids": record.get("request_ids", []),
            "execution_id": record.get("execution_id"),
            **(_measurement([record]) if record["event"] == "engine_output_send"
               else {"state": "observed", "duration_ns": None}),
        })
        for key in keys_for(record, index):
            if key not in batch["request_keys"]:
                batch["request_keys"].append(key)

    scheduler_labels = {
        "scheduler_schedule": "调度 schedule（共享外框）",
        "scheduler_kv_range": "缓存管理区间",
    }
    worker_labels = {
        "worker.execute": "执行方法外框 worker.execute",
        "runner.execute": "执行方法外框 runner.execute",
        "runner.input_prepare": "输入 / 模型准备",
        "runner.forward": "模型 forward（主机调用）",
        "runner.logits": "Logits 计算",
        "runner.sampling": "实际 sampling",
        "runner.draft": "MTP / draft",
        "runner.output_prepare": "输出准备 / 异步拷贝提交",
        "runner.postprocess": "结果后处理",
        "runner.sample_tokens": "采样方法外框（不是实际 sampling）",
        "worker.sample_tokens": "Worker 采样方法外框",
        "kv.load_submit": "KV 加载入口（非异步完成）",
        "kv.finish_forward": "KV finish_forward",
        "kv.save_wait": "KV 保存等待（独立于 sampling）",
        "kv.transfer_results": "KV 完成结果读取",
        "kv.mamba_state_copy": "Mamba 状态拷贝",
    }
    rank_evidence: dict[tuple, dict[str, set]] = {}
    for record in records:
        if (record.get("tp_rank") is None
                or any(value is None for value in clock(record))):
            continue
        worker = (*clock(record), record["tp_rank"])
        evidence = rank_evidence.setdefault(worker, {"dp_rank": set(), "pp_rank": set()})
        for field in evidence:
            if record.get(field) is not None:
                evidence[field].add(record[field])

    def worker_domain(record: dict) -> tuple:
        optional = [record.get("dp_rank"), record.get("pp_rank")]
        evidence = rank_evidence.get((*clock(record), record["tp_rank"]), {})
        # Same process lifecycle and TP are required. Conflicting optional
        # ranks leave partial records separate rather than choosing a worker.
        if evidence and all(len(values) <= 1 for values in evidence.values()):
            for index, field in enumerate(("dp_rank", "pp_rank")):
                if optional[index] is None and len(evidence[field]) == 1:
                    optional[index] = next(iter(evidence[field]))
        return (*clock(record), *optional, record["tp_rank"])

    by_key = {request["key"]: request for request in requests.values()}
    for identity, execution in executions.items():
        shared = []
        workers: dict[tuple, list[dict]] = defaultdict(list)
        member_seen = set()
        for index in execution["record_indices"]:
            record = records[index]
            for key in keys_for(record, index):
                if key not in execution["request_keys"]:
                    execution["request_keys"].append(key)
            if record.get("no_forward") is not None:
                execution["no_forward"] = record["no_forward"]
            for member in record.get("members", []):
                if not isinstance(member, dict):
                    continue
                signature = (*clock(record), str(member.get("request_id")))
                if signature not in member_seen:
                    member_seen.add(signature)
                    execution["members"].append({
                        **member, "domain": list(clock(record)),
                    })
            if record.get("tp_rank") is None:
                shared.append(record)
            else:
                rank = worker_domain(record)
                workers[rank].append(record)
        for record in records:
            if record["event"] != "request_hash_ranges":
                continue
            for part in record.get("ranges", []):
                if (
                    part.get("execution_id") is not None
                    and str(part["execution_id"]) == identity
                    and clock(record)
                    in {clock(r) for r, _ in execution_anchors.get(identity, [])}
                ):
                    shared.append({**record, **part, "event": "request_hash_call"})
        execution["shared"] = _stage_rows(
            shared,
            {
                "scheduler_schedule": scheduler_labels["scheduler_schedule"],
                "engine_execute_submit": "Engine 执行提交（非 Worker 计算完成）",
                "engine_sample_submit": "Engine 采样提交",
                "request_hash_call": "↳ hash 实际调用（显式本步身份）",
                "scheduler_external_lookup": "External KV 查询（实际调用）",
                "engine_scheduler_update": "Engine update_from_output",
            },
        )
        for record in shared:
            if record["event"] != "scheduler_schedule":
                continue
            labels = {
                "lookup": "本地缓存 lookup",
                "alloc": "KV 分配 alloc",
                "free": "KV 释放",
                "remove_skipped": "跳过块处理",
                "detach_for_deferred_free": "延迟释放摘除",
            }
            for stage, value in record.get("manager_ranges_ns", {}).items():
                execution["shared"].append(
                    {
                        "event": f"manager_ranges_ns.{stage}",
                        "label": "↳ " + labels.get(stage, stage),
                        "domain": list(clock(record)),
                        "state": ("measured" if _number(value) else "missing"),
                        "duration_ns": value if _number(value) else None,
                        "note": "生产者非重叠 manager 调用累计；不再叠加父项",
                    }
                )
            if "external_lookup_ns" in record:
                value = record["external_lookup_ns"]
                execution["shared"].append(
                    {
                        "event": "external_lookup",
                        "label": "↳ external lookup",
                        "domain": list(clock(record)),
                        "state": "measured" if _number(value) else "missing",
                        "duration_ns": value if _number(value) else None,
                        "note": "实际 connector 调用累计；含未能调度的尝试",
                    }
                )
            if "other_ns" in record:
                value = record["other_ns"]
                execution["shared"].append(
                    {
                        "event": "other_ns",
                        "label": "↳ 其他 / 未细分",
                        "domain": list(clock(record)),
                        "state": ("measured" if _number(value) else "missing"),
                        "duration_ns": value if _number(value) else None,
                        "note": record.get("other_source"),
                    }
                )
        # Per-request detail rows: scheduler_kv_range events carry request_id.
        # Batch cells above stay batch-level; these rows let the renderer
        # attribute lookup/alloc/hash to one request when it is selected.
        per_request = defaultdict(list)
        for record in shared:
            if record["event"] == "scheduler_kv_range":
                per_request[str(record.get("request_id"))].append(record)
        for request_id, items in per_request.items():
            for stage in ("lookup", "alloc", "hash"):
                stage_items = [r for r in items if r.get("stage") == stage]
                if not stage_items:
                    continue
                total = sum(
                    r.get("duration_ns", 0)
                    for r in stage_items
                    if _number(r.get("duration_ns")) is not None
                )
                execution["shared"].append(
                    {
                        "event": f"kv_range.{stage}",
                        "request_id": request_id,
                        "domain": list(clock(stage_items[0])),
                        "state": "measured",
                        "duration_ns": total,
                        "note": f"请求 {request_id} 的 {stage} 实际调用累计",
                    }
                )
        observed_stages = {stage["event"] for stage in execution["shared"]}
        for event, label in (
            ("request_hash_call", "↳ hash 实际调用（未发出本步区间）"),
            ("manager_ranges_ns.lookup", "↳ 本地缓存 lookup"),
            ("external_lookup", "↳ external lookup（无独立区间）"),
            ("manager_ranges_ns.alloc", "↳ KV 分配 alloc"),
            ("other_ns", "↳ 其他 / 未细分"),
        ):
            if event not in observed_stages:
                execution["shared"].append(
                    {
                        "event": event,
                        "label": label,
                        "state": "missing",
                        "duration_ns": None,
                        "note": "没有该阶段的实际已发出证据；未补零",
                    }
                )
        execution["scheduler_gaps"] = (
            "仅使用带原始 execution_id 的 hash ranges；其他 hash 见请求实际区间"
            "及成员累计值，不从累计值反推本步。hash 可能包含在 other 内，不能相加。"
            "external lookup 不以本地 lookup 或 Pool lookup 代替。"
        )
        execution["workers"] = [
            {"domain": list(rank), "stages": _worker_stages(group, worker_labels, rank),
             "coverage": _worker_coverage(group)}
            for rank, group in workers.items()
        ]
        # Worker-only logs still expose members without inventing token counts.
        if not execution["members"]:
            member_ids = []
            for index in execution["record_indices"]:
                for value in records[index].get("request_ids", []):
                    if value not in member_ids:
                        member_ids.append(value)
            execution["members"] = [{"request_id": value} for value in member_ids]
        for key in execution["request_keys"]:
            by_key[key]["execution_ids"].append(identity)

    tasks: dict[tuple, dict] = {}
    for index, record in enumerate(records):
        if record.get("task_id") is None:
            continue
        identity = (*domain(record, index), str(record["task_id"]))
        task = tasks.setdefault(identity, {
            "key": f"task-{len(tasks) + 1}", "task_id": str(record["task_id"]),
            "domain": list(clock(record)), "record_indices": [],
            "request_keys": [], "execution_ids": [], "stages": [],
            "members": [], "counters": [], "statuses": [], "operation": [],
            "idle": True,
        })
        task["record_indices"].append(index)
        if record.get("status") not in ("missing_load_spec", "no_task",
                                        "nothing_to_do"):
            task["idle"] = False
        for key in keys_for(record, index):
            if key not in task["request_keys"]:
                task["request_keys"].append(key)
        for field, destination in (("execution_id", "execution_ids"),
                                   ("operation", "operation"),
                                   ("status", "statuses")):
            value = record.get(field)
            if value is not None and value not in task[destination]:
                task[destination].append(value)
        if record.get("stages") or record.get("counters"):
            task["idle"] = False
        ids = record.get("request_ids", [])
        if record.get("request_id") is not None:
            ids = [record["request_id"], *ids]
        for value in ids:
            if value not in task["members"]:
                task["members"].append(value)
        if record.get("counters"):
            task["counters"].append(record["counters"])
        for name, stage in record.get("stages", {}).items():
            value = stage.get("duration_ns")
            task["stages"].append({
                "event": name, "label": {
                    "build_key": "构建 Key", "lookup": "Pool 已存在键查询",
                    "prepare": "数据 / 地址准备", "sync": "设备同步等待",
                    "backend_put": "Backend put（Pool 存储，非 P/D transfer）",
                    "backend_get": "Backend get", "save_wait": "保存完成等待",
                    "queue_submission": "保存任务提交", "wait": "任务事件等待",
                }.get(name, name),
                "state": "measured" if _number(value) else "missing",
                "duration_ns": value if _number(value) else None,
                "calls": stage.get("calls"), "exceptions": stage.get("exceptions"),
                "note": "串行调用计数器累计；区间可嵌套，不是任务总时间的可加分区",
            })
    for task in tasks.values():
        task["envelopes"] = [
            {"event": records[index]["event"], **_measurement([records[index]])}
            for index in task["record_indices"]
            if any(field in records[index]
                   for field in ("duration_ns", "start_ns", "end_ns"))
        ]
        for record in (records[i] for i in task["record_indices"]):
            if record["event"] in (
                "pd_transfer_batch_finished", "pd_transfer_task_finished",
                "pd_transfer_meta_fetch", "pd_transfer_prepare", "pd_transfer_reformat",
            ):
                task["stages"].append({
                    "event": record["event"], "label": record["event"],
                    **_measurement([record]),
                })
            value = record.get("queue_wait_ns", record.get("counters", {}).get(
                "queue_wait_ns"))
            if value is not None:
                task["stages"].append({
                    "event": "queue_wait", "label": "任务队列等待",
                    "state": "measured" if _number(value) else "missing",
                    "duration_ns": value if _number(value) else None,
                })
        for key in task["request_keys"]:
            by_key[key]["task_keys"].append(task["key"])

    overview_events = {
        "api_resolve": "req_resolve（ASGI → typed handler，通常在官方起点外）",
        "render": "模板 / render", "tokenize": "tokenizer",
        "input_process": "req_build（input_process / 输入组装）",
        "frontend_request_registered": "req_register（frontend_request_registered / 登记位置）",
        "request_id_link": "events（request_id_link / ID 关联位置）",
        "engine_submit": "req_send（engine_submit / 本地提交调用）",
        "scheduler_queue_episode": "排队（各次 episode 的有效并集）",
    }
    for request in requests.values():
        observations = [records[i] for i in request["record_indices"]]
        aliases = request["aliases"]
        preferred = next((a["value"] for kind in (
            "api_request_id", "external_req_id", "request_id", "observation_id")
                          for a in aliases if a["kind"] == kind), request["key"])
        request["label"] = preferred
        request["domains"] = list(dict.fromkeys(clock(r) for r in observations))
        request["statuses"] = [
            {"event": r["event"], "status": r.get("status"),
             "http_status": r.get("http_status"), "request_id": r.get("request_id")}
            for r in observations if r["event"] in (
                "api_terminal", "frontend_request_terminal", "request_summary",
                "scheduler_queue_summary")
        ]
        roles = []
        role_records = observations + [
            records[index] for identity in request["execution_ids"]
            for index in executions[identity]["record_indices"]
        ]
        for record in role_records:
            role = record.get("transfer_role") or record.get("kv_role")
            if role is None and record.get("role") not in (None, "unknown"):
                role = record["role"]
            if role is not None and role not in roles:
                roles.append(role)
        request["roles"] = roles
        request["overview"] = []
        for event, label in overview_events.items():
            selected = [r for r in observations if r["event"] == event]
            request["overview"].append({
                "event": event, "label": label,
                "measurements": _stage_rows(selected, {event: label})
                    or [{"state": "missing", "duration_ns": None}],
            })
        terminal = [r for r in observations
                    if r["event"] == "frontend_request_terminal"]
        request["overview"].append({
            "event": "detokenize_duration_ns", "label": "detokenizer（实际调用累计）",
            "measurements": [
                {"domain": list(clock(r)), "request_id": r.get("request_id"),
                 "calls": r.get("detokenize_calls"),
                 "state": (
                     "not_applicable" if r.get("detokenize_calls") == 0
                     else "measured" if _number(r.get("detokenize_duration_ns"))
                     else "missing"),
                 "reasons": ["detokenize_calls=0（没有执行解码调用）"],
                 "duration_ns": r.get("detokenize_duration_ns")}
                for r in terminal
            ] or [{"state": "missing", "duration_ns": None}],
        })
        for event, label in (
            ("scheduler_schedule", "schedule（共享 execution，详见下表）"),
            ("kv.load_submit", "load（分步 × TP，入口返回非传输完成）"),
            ("runner.input_prepare", "input_prepare（runner.input_prepare）"),
            ("runner.forward", "forward（分步 × TP 主机调用）"),
            ("runner.logits", "logits（runner.logits）"),
            ("runner.sampling", "sampling（实际采样，分步 × TP）"),
            ("runner.draft", "draft（runner.draft）"),
            ("runner.output_prepare", "output_build（runner.output_prepare）"),
            ("kv.save_wait", "save（等待 / Pool 任务独立展示）"),
        ):
            measurements = []
            for identity in request["execution_ids"]:
                execution = executions[identity]
                sources = execution["shared"] if event == "scheduler_schedule" else [
                    stage for worker in execution["workers"] for stage in worker["stages"]]
                for stage in sources:
                    if stage["event"] == event:
                        measurements.append({**stage, "execution_id": identity,
                                             "shared": True})
                if event == "runner.forward" and execution["no_forward"] is True:
                    if not any(m.get("execution_id") == identity for m in measurements):
                        measurements.append({
                            "execution_id": identity, "state": "not_applicable",
                            "duration_ns": None, "reasons": ["no_forward=true"],
                            "shared": True,
                        })
            request["overview"].append({
                "event": event, "label": label,
                "measurements": measurements or [{"state": "missing",
                                                   "duration_ns": None}],
            })
        order = ("api_resolve", "render", "tokenize", "input_process",
                 "frontend_request_registered", "request_id_link", "engine_submit",
                 "scheduler_queue_episode", "scheduler_schedule", "kv.load_submit",
                 "runner.input_prepare", "runner.forward", "runner.logits",
                 "runner.sampling", "runner.draft", "runner.output_prepare",
                 "detokenize_duration_ns", "kv.save_wait")
        request["overview"].sort(key=lambda stage: order.index(stage["event"]))
        request["api_boundaries"] = _stage_rows(
            observations, {"api_terminal": "API 完整响应外框（非首 token）"})
        ingress = [r for r in observations if r["event"] == "api_ingress"]
        for record in observations:
            if record["event"] not in ("api_first_generated", "engine_first_output"):
                continue
            starts = [r for r in ingress if clock(r) == clock(record)]
            measured = {"state": "missing", "duration_ns": None}
            if len(starts) == 1:
                start = starts[0].get("start_ns")
                end = record.get("timestamp_ns")
                measured = _measurement([{**record, "start_ns": start, "end_ns": end}])
            request["api_boundaries"].append({
                "label": (
                    "API 首次有效生成输出（流式服务事件，非客户端收包）"
                    if record.get("stream") is True
                    else "非流式完整响应组装（不可当 API 首 token）"
                ) if record["event"] == "api_first_generated"
                else "API 入口 → 引擎首输出（本地前端接收）",
                "event": record["event"], "domain": list(clock(record)), **measured,
            })
        request["native_summaries"] = [
            {**r, "display_ms": {
                field: value * 1000 if _number(value) else None
                for field, value in r.items() if field in (
                    "time_to_first_token", "e2e_latency", "queued_time",
                    "prefill_time", "decode_time", "mean_time_per_output_token",
                    "response_time_to_first_token", "inference_time")
            }} for r in observations if r["event"] == "request_summary"
        ]
        cache_events = [r for r in observations
                        if r["event"] == "request_cache_lookup"]
        request["metrics"] = _request_metrics(request, observations)
        request["cache_lookups"] = cache_events
        request["input_tokens"] = request["metrics"]["input_tokens"]
        request["output_tokens"] = request["metrics"]["output_tokens"]
        request["decode_ms"] = request["metrics"]["decode_ms"]
        request["decode_steps"] = request["metrics"]["decode_steps"]
        request["pd_transfer_ms"] = request["metrics"]["pd_transfer_ms"]
        request["hbm_snapshots"] = request["metrics"]["hbm_snapshots"]
        request["pd_transfers"] = request["metrics"]["pd_transfers"]
        request["hash_ranges"] = [
            {"domain": list(clock(r)), "request_id": r.get("request_id"),
             "complete": r.get("complete"), "dropped_ranges": r.get("dropped_ranges"),
             "measurement": _measurement([
                 {**part, **dict(zip(("host", "pid", "run_id"), clock(r)))}
                 for part in r.get("ranges", [])
             ])}
            for r in observations if r["event"] == "request_hash_ranges"
        ]
        request["identity_note"] = (
            "显式 observation / execution / PD offer 关联；或 request_id_link 证明"
            "随机化内部 ID，且提交与接收在输入数据中唯一时关联 EngineCoreRequest。"
            "裸 ID、跨生命周期重复或歧义接收不合并；候选步骤单列且不计入请求耗时。"
        )
        request["transport_candidates"] = [
            candidate for candidate in transport_candidates
            if request["key"] in candidate["sender_keys"]
        ]
        request["queue_boundaries"] = _queue_rows(observations)
        request["output_batches"] = [
            batch for batch in output_batches.values()
            if request["key"] in batch["request_keys"]
        ]
        request["output_endpoints"] = [
            {"domain": list(clock(r)), "output_batch_id": r.get("output_batch_id"),
             "iteration_timestamp": r.get("iteration_timestamp"),
             "timestamp_ns": r.get("timestamp_ns"), "duration_ns": None,
             "note": "官方 IterationStats wall 端点；不重新定义 TTFT 或跨域相减"}
            for r in observations if r["event"] == "api_output_endpoint"
        ]
        position_rows = _stage_rows(observations, {
            event: event for event in _POSITION_EVENTS
        })
        timer_rows = _stage_rows(observations, {
            "engine_request_init": "engine_request_init（preprocess 原调用）",
            "engine_execute_submit": "execute_submit（EngineCore 提交调用，共享批次）",
            "engine_execute_wait": "result_collect（engine_execute_wait / 回收等待，共享批次）",
            "engine_sample_call": "result_collect（engine_sample_call / sample 调用，共享批次）",
            "engine_scheduler_update": "state_update（engine_scheduler_update / 共享批次）",
            "engine_output_send": "engine_output_send（共享批次编码 + 发送调用）",
        })
        overview = {stage["event"]: stage for stage in request["overview"]}
        overview["frontend_request_registered"] = {
            **overview["frontend_request_registered"],
            "label": "req_register（ID关联 / 本地登记事件）",
            "measurements": _stage_rows(observations, {
                "frontend_request_registered": "本地登记位置",
                "request_id_link": "ID关联位置",
            }) or [{"state": "missing", "duration_ns": None}],
        }
        overview["request_hash_ranges"] = {
            "event": "request_hash_ranges", "label": "block hash（初始化 / 追加的实际区间）",
            "measurements": [
                {"domain": row["domain"], **row["measurement"]}
                for row in request["hash_ranges"]
            ] or [{"state": "missing", "duration_ns": None}],
        }
        request["mainline"] = []
        for name, label, events in (
            ("api_input", "1 · API 输入处理",
             ("api_resolve", "render", "tokenize", "input_process",
              "frontend_request_registered", "engine_submit")),
            ("core_input", "2 · EngineCore 接收 / 初始化 / 调度提交",
             ("engine_input_received", "engine_request_init", "request_hash_ranges",
              "engine_input_queue_wait", "scheduler_queue_episode",
              "scheduler_schedule", "engine_execute_submit", "engine_batch_enqueued")),
            ("worker", "3 · Worker 模型 / KV load-save",
             ("runner.input_prepare", "kv.load_submit", "runner.forward",
              "runner.logits", "runner.sampling", "runner.draft",
              "runner.output_prepare", "kv.save_wait")),
            ("core_output", "4 · EngineCore 回收 / 更新 / 回传",
             ("engine_batch_dequeued", "engine_execute_wait", "engine_sample_call",
              "engine_scheduler_update", "engine_output_queue_wait",
              "engine_output_send")),
            ("api_output", "5 · API 接收 / 官方端点",
             ("api_output_received", "api_output_endpoint")),
        ):
            stages = []
            for event in events:
                if event in overview:
                    stages.append(overview[event])
                else:
                    rows = [r for r in [
                        *position_rows, *timer_rows, *request["queue_boundaries"]
                    ] if r["event"] == event]
                    stages.append({
                        "event": event,
                        "label": rows[0].get("label", event) if rows else event,
                        "measurements": rows or [
                            {"state": "missing", "duration_ns": None}],
                    })
            request["mainline"].append({
                "name": name, "label": label, "stages": stages,
                "duration_ns": None,
                "note": "主线分组，不是可相加 TTFT 分区；共享 batch 不分摊",
            })
    return {
        "version": 1, "order": "first_observed_in_input",
        "requests": list(requests.values()), "executions": list(executions.values()),
        "tasks": list(tasks.values()), "unresolved_links": unresolved_links,
        "transport_links": transport_links,
        "transport_candidates": transport_candidates,
        "output_batches": list(output_batches.values()),
        "native_links": _native_thread_links(records, list(tasks.values())),
        "notes": [
            "全部耗时以 ms 展示；原生 summary 的秒值仅在显示层乘 1000。",
            "按输入文件顺序及本地观测顺序展示，不构造跨主机全局时间线。",
            "共享 Scheduler 仅展示一次；TP、混合 batch、Pool 任务、父子区间不可相加。",
            "缺失 ≠ 真实 0 ≠ 不适用；方法外框不是 sampling 或设备执行净时间。",
            "未采集 proxy / 客户端端到端 TTFT；日志存在不代表完整捕获。",
        ],
    }


def render_html(report: dict[str, Any]) -> str:
    """Render business stages in one request / step × TP worksheet (ms)."""

    def escape(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (list, tuple)):
            value = " / ".join(str(v) if v is not None else "?" for v in value)
        return html.escape(str(value), quote=True)

    def duration(metric: dict) -> str:
        state = metric.get("state")
        if state == "not_applicable":
            return '<span class="missing">不适用</span>'
        if state == "fused":
            return '<span class="missing">融合于 render</span>'
        if state == "observed":
            return '<span title="瞬时位置证据，不是计时器">已观测（瞬时）</span>'
        if metric.get("event") == "request_hash_call" and state == "missing":
            return (
                '<span class="missing" title="没有携带本步 execution_id 的 hash 区间；'
                '初始化 hash 可在调度前发生，缺少记录不等于耗时为零或未执行">'
                '无本步记录</span>'
            )
        value = metric.get("duration_ns")
        if state != "measured" or not _number(value):
            return (
                '<span class="missing" title="'
                + escape(metric.get("note", "此阶段没有有效计时记录"))
                + '">未采集</span>'
            )
        return f"{value / 1e6:,.6f}".rstrip("0").rstrip(".")

    evidence = []
    seen = set()
    for group in report.get("requests", []) + report.get("tasks", []):
        for record in group.get("observations", []):
            if id(record) not in seen:
                evidence.append(record)
                seen.add(id(record))
    for execution in report.get("executions", []):
        groups = [execution.get("shared_observations", [])]
        groups.extend(rank["observations"] for rank in execution.get("ranks", []))
        for group in groups:
            for record in group:
                if id(record) not in seen:
                    evidence.append(record)
                    seen.add(id(record))
    for record in report.get("unassociated", []):
        if id(record) not in seen:
            evidence.append(record)
            seen.add(id(record))
    dashboard = report.get("dashboard")
    if dashboard is None:
        dashboard = build_dashboard(evidence)

    # Paths generate the nested table header; data stays on request or TP rows.
    columns = [
        ("ttft", ("请求总计", "native TTFT")),
        ("api_resolve", ("请求入口", "req_resolve")),
        ("render", ("请求入口", "render")),
        ("tokenize", ("请求入口", "tokenizer")),
        ("input_process", ("请求入口", "组装 EngineCoreRequest")),
        ("engine_submit", ("请求入口", "发送")),
        ("scheduler_queue_episode", ("EngineCore · 调度", "排队")),
        ("scheduler_schedule", ("EngineCore · 调度", "schedule", "总耗时 · 批次共享")),
        ("request_hash_call", ("EngineCore · 调度", "block hash 计算 · 本步")),
        (
            "manager_ranges_ns.lookup",
            ("EngineCore · 调度", "schedule", "local lookup"),
        ),
        (
            "external_lookup",
            ("EngineCore · 调度", "schedule", "external lookup"),
        ),
        ("manager_ranges_ns.alloc", ("EngineCore · 调度", "schedule", "alloc")),
        ("other_ns", ("EngineCore · 调度", "schedule", "other")),
        ("engine_execute_submit", ("EngineCore · 调度", "Worker 提交")),
        ("tp_rank", ("Worker · 当前 TP", "TP")),
        ("kv.load_submit", ("Worker · 当前 TP", "start_load_kv")),
        (
            "runner.input_prepare",
            ("Worker · 当前 TP", "execute_model", "input_prepare"),
        ),
        ("runner.forward", ("Worker · 当前 TP", "execute_model", "forward")),
        ("runner.logits", ("Worker · 当前 TP", "execute_model", "logits")),
        ("kv.finish_forward", ("Worker · 当前 TP", "execute_model", "finish_forward")),
        ("runner.sampling", ("Worker · 当前 TP", "sample_tokens", "sampling")),
        ("runner.draft", ("Worker · 当前 TP", "sample_tokens", "MTP / draft")),
        (
            "runner.output_prepare",
            ("Worker · 当前 TP", "sample_tokens", "output_prepare"),
        ),
        ("runner.postprocess", ("Worker · 当前 TP", "sample_tokens", "postprocess")),
        ("kv.save_wait", ("Worker · 当前 TP", "wait_for_save", "connector 总耗时")),
        ("save.queue_submission", ("Worker · 当前 TP", "wait_for_save", "enqueue")),
        ("save.queue_wait", ("Worker · 当前 TP", "wait_for_save", "q_wait")),
        ("save.build_key", ("Worker · 当前 TP", "wait_for_save", "build_key")),
        ("save.lookup", ("Worker · 当前 TP", "wait_for_save", "lookup")),
        ("save.prepare", ("Worker · 当前 TP", "wait_for_save", "prepare")),
        ("save.sync", ("Worker · 当前 TP", "wait_for_save", "sync")),
        (
            "save.backend_put",
            ("Worker · 当前 TP", "wait_for_save", "put", "Python 总耗时"),
        ),
        (
            "save.native.phase1_start_batch_put",
            ("Worker · 当前 TP", "wait_for_save", "put", "start"),
        ),
        (
            "save.native.phase2_submit_transfers",
            ("Worker · 当前 TP", "wait_for_save", "put", "submit"),
        ),
        (
            "save.native.phase3_wait_transfers",
            ("Worker · 当前 TP", "wait_for_save", "put", "wait"),
        ),
        (
            "save.native.phase4_finalize_batch_put",
            ("Worker · 当前 TP", "wait_for_save", "put", "finalize"),
        ),
        (
            "save.native.total",
            ("Worker · 当前 TP", "wait_for_save", "put", "native total · 按 scope"),
        ),
        ("save.keys", ("Worker · 当前 TP", "wait_for_save", "keys")),
        ("save.MiB", ("Worker · 当前 TP", "wait_for_save", "MiB")),
        ("save.status", ("Worker · 当前 TP", "wait_for_save", "状态")),
        ("save.put_calls", ("Worker · 当前 TP", "wait_for_save", "put", "调用次数")),
        ("save.task", ("Worker · 当前 TP", "wait_for_save", "task ID")),
        ("engine_execute_wait", ("EngineCore · 收尾", "等待 Worker 结果")),
        ("engine_scheduler_update", ("EngineCore · 收尾", "update_from_output")),
        ("detokenize_duration_ns", ("Frontend · 请求级（非 Worker）", "detokenizer")),
    ]
    branch_order: dict[tuple, int] = {}
    for _, path in columns:
        for depth in range(1, len(path) + 1):
            branch_order.setdefault(path[:depth], len(branch_order))
    columns.sort(
        key=lambda column: tuple(
            branch_order[column[1][:depth]] for depth in range(1, len(column[1]) + 1)
        )
    )
    keys = [key for key, _ in columns]
    by_request = {request["key"]: request for request in dashboard["requests"]}
    step_numbers = {
        execution["execution_id"]: str(index + 1)
        for index, execution in enumerate(dashboard["executions"])
    }
    maintenance = [
        execution
        for execution in dashboard["executions"]
        if execution["no_forward"] is True
        and not any(
            stage["event"] == "runner.forward" and stage.get("state") == "measured"
            for worker in execution["workers"]
            for stage in worker["stages"]
        )
    ]
    maintenance_ids = {execution["execution_id"] for execution in maintenance}

    def labels(request_keys: list[str]) -> str:
        return (
            " + ".join(
                by_request[key]["label"] for key in request_keys if key in by_request
            )
            or "未关联"
        )

    def measures(rows: list[dict]) -> dict[str, str]:
        result: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            result[row["event"]].append(duration(row))
        return {event: "<br>".join(values) for event, values in result.items()}

    shared_cells: dict[str, str] = {}
    shared_span = 0

    def row(
        request_keys: list[str],
        step: str,
        tp: str,
        values: dict[str, str],
        note: str,
        kind: str,
        title: str = "",
        identity_label: str | None = None,
    ) -> str:
        applicable = {
            "request-row": {
                "ttft",
                "api_resolve",
                "render",
                "tokenize",
                "input_process",
                "request_id_link",
                "frontend_request_registered",
                "engine_submit",
                "engine_input_received",
                "engine_request_init",
                "engine_input_queue_wait",
                "request_hash_ranges",
                "scheduler_queue_episode",
                "detokenize_duration_ns",
                "engine_output_enqueued",
                "engine_output_send",
                "api_output_received",
                "api_output_endpoint",
            },
            "shared-row": {
                "scheduler_schedule",
                "request_hash_call",
                "manager_ranges_ns.lookup",
                "external_lookup",
                "manager_ranges_ns.alloc",
                "other_ns",
                "engine_execute_submit",
                "engine_execute_wait",
                "engine_scheduler_update",
            },
            "worker-row": {
                key for key in keys if key.startswith(("runner.", "kv.", "save."))
            },
        }.get(kind, set(values))
        if kind == "worker-row":
            applicable.update(("runner.execute", "runner.sample_tokens"))
        empty = {
            key: (
                '<span class="missing">未采集</span>'
                if key in applicable
                else '<span class="missing" title="不属于此行统计范围">—</span>'
            )
            for key in keys
        }
        cells = []
        identity_span = shared_span if kind == "worker-row" else 1
        for key in keys:
            if key == "tp_rank":
                cells.append(
                    '<td class="tp" data-stage="tp_rank">' + escape(tp) + "</td>"
                )
                continue
            if kind == "worker-row" and key in shared_cells:
                if shared_span:
                    cells.append(
                        f'<td rowspan="{shared_span}" data-stage="{escape(key)}">'
                        + shared_cells[key]
                        + "</td>"
                    )
                continue
            cells.append(
                f'<td data-stage="{escape(key)}">'
                + values.get(key, empty[key])
                + "</td>"
            )
        return (
            '<tr class="'
            + kind
            + '" data-requests="'
            + escape(" ".join(request_keys))
            + '" title="'
            + escape(title)
            + '">'
            + (
                '<td class="request" rowspan="'
                + str(identity_span)
                + '">'
                + escape(
                    identity_label
                    if identity_label is not None
                    else labels(request_keys)
                )
                + '</td><td class="step" rowspan="'
                + str(identity_span)
                + '">'
                + escape(step)
                + "</td>"
                if identity_span
                else ""
            )
            + "".join(cells)
            + '<td class="note">'
            + escape(note)
            + "</td></tr>"
        )

    header_depth = max(len(path) for _, path in columns)
    headers = [[] for _ in range(header_depth)]
    for depth in range(header_depth):
        index = 0
        while index < len(columns):
            path = columns[index][1]
            if len(path) <= depth:
                index += 1
                continue
            end = index + 1
            while (
                end < len(columns) and columns[end][1][: depth + 1] == path[: depth + 1]
            ):
                end += 1
            rowspan = header_depth - depth if len(path) == depth + 1 else 1
            headers[depth].append(
                f'<th class="l{min(depth + 1, 3)}" colspan="{end - index}" '
                f'rowspan="{rowspan}" data-stage="'
                + escape(columns[index][0] if len(path) == depth + 1 else "")
                + '" data-path="'
                + escape(" / ".join(path[: depth + 1]))
                + '">'
                + ('<span class="process-label">' if depth == 0 else "")
                + escape(path[depth])
                + ("</span>" if depth == 0 else "")
                + "</th>"
            )
            index = end
    pinned = (
        f'<th class="request" rowspan="{header_depth}">请求</th>'
        f'<th class="step" rowspan="{header_depth}">step</th>'
    )
    head = "<tr>" + pinned + "".join(headers[0])
    head += f'<th rowspan="{header_depth}">批次成员 / 说明</th></tr>'
    head += "".join("<tr>" + "".join(h) + "</tr>" for h in headers[1:])
    body = []
    for request in dashboard["requests"]:
        values = {}
        for stage in request["overview"]:
            if stage["event"] in {
                "api_resolve",
                "render",
                "tokenize",
                "input_process",
                "request_id_link",
                "frontend_request_registered",
                "engine_submit",
                "scheduler_queue_episode",
                "detokenize_duration_ns",
            }:
                values[stage["event"]] = "<br>".join(
                    duration(m) for m in stage["measurements"]
                )
        summaries = request["native_summaries"]
        values["ttft"] = (
            "<br>".join(
                "不适用（未观测首 token）"
                if s.get("first_token_observed") is False
                else duration(
                    {
                        "state": "measured",
                        "duration_ns": s["display_ms"]["time_to_first_token"] * 1e6,
                    }
                )
                if s.get("first_token_observed") is True
                and _number(s["display_ms"].get("time_to_first_token"))
                else "—"
                for s in summaries
            )
            or "—"
        )
        note = "; ".join(
            f"输入 {s.get('num_prompt_tokens', '?')} / "
            f"输出 {s.get('num_generation_tokens', '?')} / {s.get('status', '?')}"
            for s in summaries
        )
        if request.get("candidate_for"):
            note = "关联待确认；不计入其他请求"
        for phase in request["mainline"]:
            for stage in phase["stages"]:
                event = stage["event"]
                if (
                    event in keys
                    and event not in values
                    and event
                    not in {
                        "scheduler_schedule",
                        "engine_execute_submit",
                        "engine_execute_wait",
                        "engine_scheduler_update",
                        "kv.load_submit",
                        "kv.save_wait",
                    }
                    and not event.startswith("runner.")
                ):
                    values[event] = "<br>".join(
                        duration(item) for item in stage["measurements"]
                    )
        for batch in request["output_batches"]:
            for item in batch["evidence"]:
                event = item["event"]
                if event in keys and event not in values:
                    values[event] = duration(item)
        body.append(
            row(
                [request["key"]],
                "请求级",
                "—",
                values,
                note,
                "request-row",
                request["identity_note"],
            )
        )

    shown_tasks = set()

    def save_values(execution: dict, worker: dict) -> dict[str, str]:
        domain = worker["domain"]
        result: dict[str, list[str]] = defaultdict(list)
        for task in dashboard["tasks"]:
            if task["execution_ids"] != [execution["execution_id"]]:
                continue
            if task["domain"] != domain[:3]:
                continue
            records = [
                record
                for record in evidence
                if str(record.get("task_id")) == task["task_id"]
                and list(clock(record)) == task["domain"]
            ]
            ranks = {
                record.get("tp_rank")
                for record in records
                if record.get("tp_rank") is not None
            }
            if ranks != {domain[-1]}:
                continue
            candidates = [
                candidate
                for candidate in execution["workers"]
                if candidate["domain"][:3] == task["domain"]
                and candidate["domain"][-1] == domain[-1]
            ]
            if len(candidates) != 1:
                continue
            if any(
                record.get(field) is not None and record[field] != domain[index]
                for record in records
                for field, index in (("dp_rank", 3), ("pp_rank", 4))
            ):
                continue
            if not any(
                "save" in operation or operation == "mooncake_put"
                for operation in task["operation"]
            ):
                continue
            shown_tasks.add(task["key"])
            is_submission = set(task["operation"]) == {"save_submit"}
            if is_submission and "no_task" in task["statuses"] and not task["stages"]:
                for key in keys:
                    if key.startswith("save."):
                        result[key] = [
                            "未执行保存（无任务）"
                            if key == "save.status"
                            else "—"
                        ]
                continue
            values = measures(task["stages"])
            is_submission = set(task["operation"]) == {"save_submit"}
            if is_submission:
                values = {
                    key: value
                    for key, value in values.items()
                    if key == "queue_submission"
                }
                if not values and "no_task" in task["statuses"]:
                    values["status"] = "未执行保存（无任务）"
                    values["task"] = "—"
                elif values:
                    submit_states = {
                        "submission_returned": "已提交（后台结果独立记录）",
                        "no_task": "未执行保存（无任务）",
                    }
                    values["status"] = escape(
                        "/".join(submit_states.get(state, state)
                                 for state in task["statuses"])
                        or "提交状态未知"
                    )
                    values["task"] = escape(task["task_id"])
            else:
                states = {
                    "no_task": "未执行保存（无任务）",
                    "all_hit": "查重全命中（未 PUT）",
                    "succeeded": "保存成功",
                    "backend_failure": "保存失败",
                }
                values["status"] = escape(
                    "/".join(states.get(state, state) for state in task["statuses"])
                    or "状态未知"
                )
                values["task"] = escape(task["task_id"])
            no_work = (
                bool(task["statuses"])
                and set(task["statuses"]) <= {"no_task", "all_hit"}
                and not any(
                    stage["event"].startswith("backend_") for stage in task["stages"]
                )
            )
            for field, divisor in (("keys", 1), ("MiB", 1048576)):
                suffix = "submitted_keys" if field == "keys" else "submitted_bytes"
                counts = []
                for counter in task["counters"]:
                    for operation in (
                        "put",
                        "batch_put_from_multi_buffer_ranges",
                        "mooncake_put",
                    ):
                        value = counter.get(operation + "_" + suffix)
                        if _number(value):
                            counts.append(f"{value / divisor:g}")
                            break
                if counts:
                    values[field] = "<br>".join(counts)
            links = [
                link
                for link in dashboard.get("native_links", [])
                if link.get("task_key") == task["key"]
                and link.get("call_record_index") is not None
            ]
            calls: dict[int, list[dict]] = defaultdict(list)
            for link in links:
                calls[link["call_record_index"]].append(link["native"])
            if calls:
                values["put_calls"] = str(len(calls))
            for field in (
                "phase1_start_batch_put",
                "phase2_submit_transfers",
                "phase3_wait_transfers",
                "phase4_finalize_batch_put",
                "total",
            ):
                durations = []
                details = []
                for number, natives in enumerate(calls.values(), 1):
                    observed = [
                        native
                        for native in natives
                        if field in native.get("duration_fields_ns", {})
                    ]
                    # Split scopes fill different phases of one call. Repeated
                    # measurements of the same phase cannot be added or chosen.
                    value = (
                        observed[0]["duration_fields_ns"][field]
                        if len(observed) == 1
                        else None
                    )
                    durations.append(value)
                    scope = "/".join(
                        native.get("timing_scope", "unspecified") for native in observed
                    )
                    details.append(f"PUT #{number} · {scope}: {value} ns")
                if calls:
                    complete = all(_number(value) for value in durations)
                    if field == "total":
                        scopes = {
                            native.get("timing_scope", "unspecified")
                            for natives in calls.values()
                            for native in natives
                            if "total" in native.get("duration_fields_ns", {})
                        }
                        complete = complete and len(scopes) == 1
                    metric = {
                        "state": "measured" if complete else "missing",
                        "duration_ns": sum(durations) if complete else None,
                    }
                    label = f"{len(calls)} 次 PUT；同阶段累计；" + "; ".join(details)
                    values["native." + field] = (
                        '<span title="'
                        + escape(label)
                        + '">'
                        + duration(metric)
                        + "</span>"
                    )
            if no_work:
                for key in keys:
                    field = key.removeprefix("save.")
                    if key.startswith("save.") and field not in values:
                        values[field] = "—"
            for field, value in values.items():
                key = "save." + field
                if key not in keys:
                    continue
                if is_submission and field != "queue_submission":
                    continue
                if not is_submission and field == "queue_submission":
                    continue
                if field in {"status", "task", "keys", "MiB", "put_calls"}:
                    if not is_submission:
                        result[key] = [value]
                    continue
                title = "" if no_work or values.get("task") == "—" else escape(task["task_id"])
                result[key].append('<span title="' + title + '">' + value + '</span>')
        return {key: "<br>".join(values) for key, values in result.items()}

    for execution in dashboard["executions"]:
        if execution["execution_id"] in maintenance_ids:
            continue
        step = step_numbers[execution["execution_id"]]
        members = "; ".join(
            f"{m['request_id']}: {m.get('scheduled_tokens', '?')} tokens"
            for m in execution["members"]
        )
        note = members + ("; no_forward" if execution["no_forward"] is True else "")
        if execution.get("completion_request_ids"):
            note += "; KV 完成通知：" + ", ".join(execution["completion_request_ids"])
        if not execution["request_keys"]:
            domain = (
                execution["workers"][0]["domain"][:3] if execution["workers"] else None
            )
            note += "; 无请求成员证据；Worker 进程：" + str(domain)
        shared_cells = measures(execution["shared"])
        shared_cells.update(
            measures(
                _stage_rows(
                    [
                        record
                        for record in evidence
                        if record.get("execution_id") == execution["execution_id"]
                        and record.get("tp_rank") is None
                    ],
                    {"engine_execute_wait": "等待 Worker 结果"},
                )
            )
        )
        for key in (
            "scheduler_schedule",
            "request_hash_call",
            "manager_ranges_ns.lookup",
            "external_lookup",
            "manager_ranges_ns.alloc",
            "other_ns",
            "engine_execute_submit",
            "engine_execute_wait",
            "engine_scheduler_update",
        ):
            shared_cells.setdefault(key, '<span class="missing">未采集</span>')
        # Multi-request batches: detail cells carry per-request values so a
        # selected request sees only its own lookups/allocs; the batch total
        # moves to the tooltip. Single-request batches keep batch values
        # (identical to the per-request values there).
        member_requests = {str(m.get("request_id")) for m in execution["members"]} - {
            "None"
        }
        if len(member_requests) > 1:

            def per_request_cell(
                stage: str, batch_value: str, shared=execution["shared"]
            ) -> str:
                rows = [s for s in shared if s["event"] == f"kv_range.{stage}"]
                if not rows:
                    return batch_value
                per_req = "<br>".join(
                    duration({"state": "measured", "duration_ns": s["duration_ns"]})
                    + f' <span class="missing">({escape(s["request_id"])})</span>'
                    for s in sorted(rows, key=lambda s: str(s["request_id"]))
                )
                return (
                    f'<span title="批级合计（含其他请求）：{escape(batch_value)}">'
                    + per_req
                    + "</span>"
                )

            shared_cells["manager_ranges_ns.lookup"] = per_request_cell(
                "lookup", shared_cells.get("manager_ranges_ns.lookup", "—")
            )
            shared_cells["manager_ranges_ns.alloc"] = per_request_cell(
                "alloc", shared_cells.get("manager_ranges_ns.alloc", "—")
            )
        workers = sorted(
            execution["workers"],
            key=lambda w: (
                str(w["domain"][3:5]),
                w["domain"][-1] if isinstance(w["domain"][-1], int) else -1,
                str(w["domain"][:3]),
            ),
        )
        # Schedule and Core observations are shared across this step's TP rows.
        if not execution["workers"]:
            body.append(
                row(
                    execution["request_keys"],
                    step,
                    "共享",
                    shared_cells,
                    note,
                    "shared-row",
                    execution["execution_id"],
                )
            )
        shared_span = len(workers)
        for key, path in columns:
            if path[0] != "Worker · 当前 TP":
                shared_cells.setdefault(key, '<span class="missing">—</span>')
        for worker in workers:
            domain = worker["domain"]
            values = measures(worker["stages"])
            values.update(save_values(execution, worker))
            if execution["no_forward"] is True and "runner.forward" not in values:
                values["runner.forward"] = "不执行"
            body.append(
                row(
                    execution["request_keys"],
                    step,
                    "?" if domain[-1] is None else str(domain[-1]),
                    values,
                    note + f"; DP {domain[3]} / PP {domain[4]}",
                    "worker-row",
                    str(domain) + "; " + execution["execution_id"],
                )
            )
            shared_span = 0
        shared_cells = {}

    pending_tasks = [
        task for task in dashboard["tasks"] if task["key"] not in shown_tasks
    ]
    pending_native = [
        link
        for link in dashboard.get("native_links", [])
        if link.get("task_key") not in shown_tasks
        or link.get("call_record_index") is None
    ]
    out = [
        (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>TTFT · step × TP</title><style>"
            "body{font:14px/1.5 system-ui,sans-serif;margin:20px;"
            "color:#213047;background:#fff}"
            "h1{font-size:22px;margin:0 0 8px}p{margin:8px 0;color:#536174}"
            "select{padding:6px;max-width:80%}.sheet{overflow:auto;max-height:80vh;margin-top:12px}"
            "table{border-collapse:separate;border-spacing:0;font-size:13px;white-space:nowrap}"
            "th,td{border-right:1px solid #d5dfeb;border-bottom:1px solid #d5dfeb;"
            "padding:7px 6px;text-align:right;min-width:54px}"
            "thead{position:sticky;top:0;z-index:4}th{text-align:center;color:#182635}"
            "th.l1{background:#edf4ff;text-align:left}th.l2{background:#eef7f1}"
            "th.l3{background:#fff4df}"
            ".process-label{position:sticky;left:255px;display:inline-block}"
            ".request,.step{position:sticky;z-index:2;background:#fff;text-align:left}"
            ".request{left:0;min-width:120px;max-width:120px;overflow:hidden;text-overflow:ellipsis}"
            ".step{left:133px;min-width:50px;max-width:50px;text-align:center}"
            ".tp{min-width:36px;text-align:center;font-weight:bold}"
            "thead .request,thead .step,thead .tp{background:#f5f8fa;z-index:5}"
            ".request-row{background:#eaf1fa;font-weight:600}.shared-row{background:#f0f3f6;color:#3f4a55;font-weight:600}"
            ".worker-row{background:#fff}.worker-row:nth-of-type(even){background:#f7f9fb}"
            ".note{text-align:left;max-width:350px;white-space:normal;min-width:200px}"
            ".missing{color:#c1c8cf}"
            'td[data-stage^="save."]{background:#f4faf6}'
            "td[rowspan]{vertical-align:top;background:#f1f5fa}"
            ".worker-row:hover td{background:#eff8f4}"
            "details{margin-top:16px}"
            "summary{cursor:pointer;font-weight:650;padding:7px 0}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:60vh;overflow:auto}"
            "[hidden]{display:none!important}</style></head><body>"
            "<h1>TTFT · 请求 / step × TP</h1>"
            "<p>单位 ms · 请求总计一行，每个 step 按 TP 展开；"
            "schedule / Core 共享单元格跨 TP 合并。</p>"
            "<p><strong>普通 step 的控制顺序：</strong>API Server → EngineCore.schedule"
            " → Worker.execute_model → 必要时 EngineCore 调用 sample_tokens"
            " → Worker.sample_tokens（采样 / MTP）→ EngineCore.update_from_output"
            " → API Server 输出处理。批队列路径可重叠不同 step。</p>"
            '<label for="request-select">请求：</label><select id="request-select">'
            '<option value="all">全部请求</option>'
        ),
    ]
    for request in dashboard["requests"]:
        out.append(
            '<option value="'
            + escape(request["key"])
            + '">'
            + escape(request["label"])
            + "</option>"
        )
    out.append(
        "</select><p>wait_for_save 在同一 TP 行内横向展开 "
        "q_wait / build_key / lookup / prepare / sync / put。"
        "put 下分 start / submit / wait / finalize；keys / MiB 为当前任务提交量。</p>"
        '<div class="sheet" id="step-details"><table id="step-table"><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    metric_columns = {
        "input_tokens": "输入 tokens", "output_tokens": "输出 tokens",
        "cached_tokens": "KVC 命中 tokens（官方）",
        "hbm_cached_tokens": "HBM 命中 tokens",
        "external_cached_tokens": "外部命中 tokens",
        "prefill_compute_tokens": "需计算 prompt tokens",
        "decode_ms": "decode 总时间 ms", "decode_steps": "decode 总步数",
        "observed_decode_steps": "日志观测 decode 步数",
    }
    out.append('<h2>请求级容量 / 缓存 / decode</h2><table id="request-metrics"><thead>'
               '<tr><th>请求 ID</th>' + ''.join(
                   '<th>' + escape(label) + '</th>'
                   for label in metric_columns.values())
               + '<th>HBM / P-D 传输明细</th></tr></thead><tbody>')
    for request in dashboard["requests"]:
        metrics = request.get("metrics", {})
        out.append('<tr data-requests="' + escape(request["key"]) + '"><td>'
                   + escape(request["label"]) + '</td>')
        for name in metric_columns:
            metric = metrics.get(name, {})
            value = metric.get("value")
            text = str(value) if value is not None else "未采集"
            note = metric.get("note") or metric.get("source", "")
            out.append('<td title="' + escape(note)
                       + '">' + escape(text) + '</td>')
        detail = {name: metrics.get(name) for name in
                  ("cache_source", "hbm_snapshots", "pd_transfers")}
        out.append('<td><details><summary>按组 / TP / 调用查看</summary><pre>'
                   + escape(json.dumps(detail, ensure_ascii=False, indent=2))
                   + '</pre></details></td></tr>')
    out.append('</tbody></table>')
    native_count = len(report.get("native_timings", []))
    out.append(
        "<p>采集检查：external lookup 实际事件 "
        + str(report.get("event_counts", {}).get("scheduler_external_lookup", 0))
        + " 条；Engine update "
        + str(report.get("event_counts", {}).get("engine_scheduler_update", 0))
        + " 条；Mooncake 原生 "
        + str(native_count)
        + " 条。"
        + (
            "原生记录按同文件 TID 恢复线程归属；没有调用边界时不猜具体 put。"
            if native_count
            else "输入中没有 BatchPutTiming/BatchUpsertTiming，不能填 phase1–4。"
        )
        + "</p>"
    )
    out.append(
        "<details><summary>表格口径与异步任务说明</summary><p>"
        "schedule 是共享批次范围；wait_for_save 是当前 TP 的 connector 边界。"
        "后台 task 与 Mooncake phases 横向展示在同一 TP 行；"
        "同一 task 的 native 同阶段按调用累计，悬停查看调用次数、各次值和 scope。"
        "native phases 不与 task 外框或 wait_for_save 相加。"
        "主表只展示本步 hash；明确无 forward 的维护轮次默认折叠，不按耗时过滤。"
        "start_load_kv 是入口调用计时，返回不证明搬运完成。"
        "keys 是 API 提交计数；MiB 是提交载荷，不是网络实测。"
        "未关联的 native 记录只保留证据，不按邻近时间强行归属。"
        "— 表示不属于该层；未采集表示该层没有有效计时；零是实测零。"
        "</p></details>"
    )
    if maintenance:
        out.append(
            '<details id="maintenance-steps"><summary>维护轮次（未执行 forward）：'
            + str(len(maintenance))
            + "</summary><pre>"
            + escape(json.dumps(maintenance, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    if report.get("parse_errors"):
        out.append("<p>部分日志解析失败，详见末尾证据。</p>")
    if not dashboard["requests"]:
        out.append("<p>没有可关联的请求。</p>")
    if pending_tasks:
        out.append(
            '<details id="pending-tasks"><summary>'
            "未能唯一填入 step × TP 的任务证据</summary><pre>"
            + escape(json.dumps(pending_tasks, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    if pending_native:
        out.append(
            '<details id="pending-native"><summary>Mooncake 原生记录（'
            + str(len(pending_native))
            + " 条；未能唯一关联到任务）</summary>"
            "<p>无法唯一绑定的原生记录不计入请求或任务；线程候选和调用边界状态仅作为排查证据。</p><pre>"
            + escape(json.dumps(pending_native, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    out.append(
        '<details id="raw-evidence"><summary>口径与原始证据（需要时展开）'
        "</summary><p>step 按首次观测顺序编号；共享 batch 不分摊。"
        "任务只显示一次；跨 step 任务列出全部原始 step。"
        "load 入口返回不等于搬运完成，put 字节数不是实测网络流量。"
        "未关联的 Mooncake native phase 不按时间邻近硬配。</p><pre>"
    )
    out.append(escape(json.dumps(report, ensure_ascii=False, indent=2)))
    out.append(
        "</pre></details><script>"
        'const selector=document.getElementById("request-select");'
        "function selectRequest(){const key=selector.value;"
        'document.querySelectorAll("[data-requests]").forEach(function(el){'
        'const keys=el.dataset.requests.split(" ");'
        'el.hidden=key!=="all"&&!keys.includes(key);});}'
        'selector.addEventListener("change",selectRequest);'
        "</script></body></html>"
    )
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()
    records, errors = read_records(args.logs)
    report = summarize(records, errors)
    args.out.write_text(
        render_html(report)
        if args.html
        else json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{len(records)} observations; {len(report['executions'])} executions; "
        f"{len(errors)} invalid records -> {args.out}"
    )
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
