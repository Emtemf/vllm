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


def _measurement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One clock/event only; never turn partial evidence into a complete total."""
    result: dict[str, Any] = {"state": "missing", "duration_ns": None,
                              "observations": len(records)}
    if not records:
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
         **_measurement(group)}
        for key, group in groups.items()
    ]

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
            owner["task_candidates"] = list(
                dict.fromkeys(task_by_record[i] for i in indices if i in task_by_record)
            )
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
        if call_index is not None:
            owner = next(
                candidate
                for candidate in candidates
                if candidate["domain"] == list(clock(records[call_index]))
            )
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
                "task_key": task_by_record.get(call_index),
            }
        )
    return links


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

    # Only request-specific fields join aliases. Batch members never join.
    for index, record in enumerate(records):
        nodes = []
        for kind in ("observation_id", "api_request_id", "request_id",
                     "external_req_id"):
            value = record.get(kind)
            if value is not None:
                nodes.append(node(kind, value, record, index))
        join(nodes)
        if record["event"] == "request_child_link":
            ancestor = node("request_id", record.get("parent_request_id"),
                            record, index)
            join(nodes + ([ancestor] if ancestor is not None else []))
        record_nodes.append(nodes)

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
        ids = list(record.get("request_ids", []))
        for field in ("finished_sending", "finished_recving", "failed_recving"):
            ids.extend(record.get(field, []))
        for value in ids:
            nodes.extend(execution_nodes(record.get("execution_id"), value))
        for member in record.get("members", []):
            if isinstance(member, dict):
                nodes.extend(execution_nodes(record.get("execution_id"),
                                             member.get("request_id")))
        return list(dict.fromkeys(requests[root(n)]["key"] for n in nodes))

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
                if (part.get("execution_id") is not None
                        and str(part["execution_id"]) == identity
                        and clock(record) in {clock(r) for r, _ in
                                              execution_anchors.get(identity, [])}):
                    shared.append({**record, **part, "event": "request_hash_call"})
        execution["shared"] = _stage_rows(shared, {
            "scheduler_schedule": scheduler_labels["scheduler_schedule"],
            "engine_execute_submit": "Engine 执行提交（非 Worker 计算完成）",
            "engine_sample_submit": "Engine 采样提交",
            "request_hash_call": "↳ hash 实际调用（显式本步身份）",
            "scheduler_external_lookup": "External KV 查询（实际调用）",
            "engine_scheduler_update": "Engine update_from_output",
        })
        for record in shared:
            if record["event"] != "scheduler_schedule":
                continue
            labels = {"lookup": "本地缓存 lookup", "alloc": "KV 分配 alloc",
                      "free": "KV 释放", "remove_skipped": "跳过块处理",
                      "detach_for_deferred_free": "延迟释放摘除"}
            for stage, value in record.get("manager_ranges_ns", {}).items():
                execution["shared"].append({
                    "event": f"manager_ranges_ns.{stage}",
                    "label": "↳ " + labels.get(stage, stage),
                    "domain": list(clock(record)), "state": (
                        "measured" if _number(value) else "missing"),
                    "duration_ns": value if _number(value) else None,
                    "note": "生产者非重叠 manager 调用累计；不再叠加父项",
                })
            if "external_lookup_ns" in record:
                value = record["external_lookup_ns"]
                execution["shared"].append({
                    "event": "external_lookup", "label": "↳ external lookup",
                    "domain": list(clock(record)),
                    "state": "measured" if _number(value) else "missing",
                    "duration_ns": value if _number(value) else None,
                    "note": "实际 connector 调用累计；含未能调度的尝试",
                })
            if "other_ns" in record:
                value = record["other_ns"]
                execution["shared"].append({
                    "event": "other_ns", "label": "↳ 其他 / 未细分",
                    "domain": list(clock(record)), "state": (
                        "measured" if _number(value) else "missing"),
                    "duration_ns": value if _number(value) else None,
                    "note": record.get("other_source"),
                })
        observed_stages = {stage["event"] for stage in execution["shared"]}
        for event, label in (
            ("request_hash_call", "↳ hash 实际调用（未发出本步区间）"),
            ("manager_ranges_ns.lookup", "↳ 本地缓存 lookup"),
            ("external_lookup", "↳ external lookup（无独立区间）"),
            ("manager_ranges_ns.alloc", "↳ KV 分配 alloc"),
            ("other_ns", "↳ 其他 / 未细分"),
        ):
            if event not in observed_stages:
                execution["shared"].append({
                    "event": event, "label": label, "state": "missing",
                    "duration_ns": None, "note": "没有该阶段的实际已发出证据；未补零",
                })
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
        "api_resolve": "req_resolve / API 解析校验",
        "render": "模板 / render", "tokenize": "tokenizer",
        "engine_submit": "add_request / 本地提交调用",
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
            ("runner.forward", "forward（分步 × TP 主机调用）"),
            ("runner.sampling", "sampling（实际采样，分步 × TP）"),
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
            if event == "kv.save_wait":
                for task in tasks.values():
                    if task["key"] not in request["task_keys"]:
                        continue
                    if not any("save" in operation for operation in task["operation"]):
                        continue
                    measurements.extend({
                        **envelope, "domain": task["domain"],
                        "task_key": task["key"], "shared": len(task["members"]) > 1,
                        "note": "独立 Pool 保存任务外框，不与 Worker save wait 相加",
                    } for envelope in task["envelopes"])
            request["overview"].append({
                "event": event, "label": label,
                "measurements": measurements or [{"state": "missing",
                                                   "duration_ns": None}],
            })
        order = ("api_resolve", "render", "tokenize", "engine_submit",
                 "scheduler_queue_episode", "scheduler_schedule", "kv.load_submit",
                 "runner.forward", "runner.sampling", "detokenize_duration_ns",
                 "kv.save_wait")
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
    return {
        "version": 1, "order": "first_observed_in_input",
        "requests": list(requests.values()), "executions": list(executions.values()),
        "tasks": list(tasks.values()), "unresolved_links": unresolved_links,
        "transport_links": transport_links,
        "transport_candidates": transport_candidates,
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

    # Fixed business workflow: missing observations leave cells missing; they
    # never create an alternative call-stack hierarchy.
    columns = [
        ("ttft", ("请求总计", "native TTFT")),
        ("api_resolve", ("请求入口", "req_resolve")),
        ("render", ("请求入口", "render")),
        ("tokenize", ("请求入口", "tokenizer")),
        ("engine_submit", ("请求入口", "add_request")),
        ("scheduler_queue_episode", ("Scheduler · execution 共享", "排队")),
        ("request_hash_call", ("Scheduler · execution 共享", "block hash")),
        ("scheduler_schedule", ("Scheduler · execution 共享", "schedule")),
        ("manager_ranges_ns.lookup", ("Scheduler · execution 共享", "local lookup")),
        ("external_lookup", ("Scheduler · execution 共享", "external lookup")),
        ("manager_ranges_ns.alloc", ("Scheduler · execution 共享", "alloc")),
        ("other_ns", ("Scheduler · execution 共享", "other")),
        ("kv.load_submit", ("Worker · 当前 TP", "start_load_kv")),
        ("runner.execute", ("Worker · 当前 TP", "forward / 执行", "runner execute")),
        (
            "runner.input_prepare",
            ("Worker · 当前 TP", "forward / 执行", "input_prepare"),
        ),
        ("runner.forward", ("Worker · 当前 TP", "forward / 执行", "forward")),
        ("runner.logits", ("Worker · 当前 TP", "forward / 执行", "logits")),
        (
            "kv.mamba_state_copy",
            ("Worker · 当前 TP", "forward / 执行", "Mamba state copy"),
        ),
        ("kv.finish_forward", ("Worker · 当前 TP", "保存入口", "finish_forward")),
        ("worker.sample_tokens", ("Worker · sampling / MTP", "sample_tokens")),
        ("runner.sample_tokens", ("Worker · sampling / MTP", "runner sample")),
        ("runner.sampling", ("Worker · sampling / MTP", "sampling")),
        ("runner.draft", ("Worker · sampling / MTP", "MTP / draft")),
        ("runner.output_prepare", ("Worker · sampling / MTP", "output_prepare")),
        ("runner.postprocess", ("Worker · sampling / MTP", "postprocess")),
        ("kv.transfer_results", ("Worker · sampling / MTP", "get_transfer_results")),
        ("kv.save_wait", ("wait_for_save / 保存相关明细", "等待接口")),
        ("pool_save_fence", ("wait_for_save / 保存相关明细", "释放前 fence")),
        ("task_total", ("wait_for_save / 保存相关明细", "独立任务外框")),
        ("queue_submission", ("wait_for_save / 保存相关明细", "enqueue")),
        ("queue_wait", ("wait_for_save / 保存相关明细", "q_wait")),
        ("build_key", ("wait_for_save / 保存相关明细", "build_key")),
        ("lookup", ("wait_for_save / 保存相关明细", "lookup")),
        ("prepare", ("wait_for_save / 保存相关明细", "prepare")),
        ("sync", ("wait_for_save / 保存相关明细", "sync")),
        ("backend_put", ("wait_for_save / 保存相关明细", "put", "Backend.put")),
        (
            "backend_mooncake_put",
            ("wait_for_save / 保存相关明细", "put", "Mooncake Python 调用"),
        ),
        ("phase1_start_batch_put", ("wait_for_save / 保存相关明细", "put", "start")),
        ("phase2_submit_transfers", ("wait_for_save / 保存相关明细", "put", "submit")),
        ("phase3_wait_transfers", ("wait_for_save / 保存相关明细", "put", "wait")),
        (
            "phase4_finalize_batch_put",
            ("wait_for_save / 保存相关明细", "put", "finalize"),
        ),
        ("native_total", ("wait_for_save / 保存相关明细", "put", "total · 原记录")),
        ("task_keys", ("wait_for_save / 保存相关明细", "keys · 提交")),
        ("task_mib", ("wait_for_save / 保存相关明细", "MiB · 提交载荷")),
        ("task_counters", ("wait_for_save / 保存相关明细", "结果计数")),
        ("task_status", ("wait_for_save / 保存相关明细", "结果")),
        ("engine_scheduler_update", ("Engine", "update_from_output")),
        ("detokenize_duration_ns", ("Frontend · 请求级（非 Worker）", "detokenizer")),
    ]
    for phase in ("execute", "sample"):
        for field, label in (
            ("duration_ns", "实际耗时"),
            ("covered_ns", "已覆盖 union"),
            ("uncovered_ns", "局部未覆盖"),
        ):
            columns.append(
                (
                    f"coverage.{phase}.{field}",
                    ("Worker-local coverage · 当前 TP", phase, label),
                )
            )
    columns.extend(
        [
            ("backend_get", ("Pool load · 非 put", "Backend.get")),
            ("backend_mooncake_get", ("Pool load · 非 put", "Mooncake get")),
            ("pd_transfer_task_finished", ("P/D read · 非 Pool put", "任务外框")),
            ("pd_queue_wait", ("P/D read · 非 Pool put", "q_wait")),
            ("pd_transfer_meta_fetch", ("P/D read · 非 Pool put", "metadata fetch")),
            ("pd_transfer_prepare", ("P/D read · 非 Pool put", "prepare")),
            (
                "pd_transfer_batch_finished",
                ("P/D read · 非 Pool put", "batch_transfer_sync_read"),
            ),
            ("pd_transfer_reformat", ("P/D read · 非 Pool put", "reformat")),
        ]
    )
    session_events = sorted(
        {
            stage["event"]
            for task in dashboard["tasks"]
            for stage in task["stages"]
            if stage["event"].startswith("backend_batch_")
        }
    )
    columns.extend(
        (
            event,
            (
                "Pool load · 非 put"
                if event.startswith("backend_batch_get")
                else "wait_for_save / 保存相关明细",
                "get" if event.startswith("backend_batch_get") else "put",
                event.removeprefix("backend_"),
            ),
        )
        for event in session_events
    )
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
    by_request = {r["key"]: r for r in dashboard["requests"]}
    step_numbers = {
        e["execution_id"]: str(i + 1) for i, e in enumerate(dashboard["executions"])
    }

    def labels(request_keys: list[str]) -> str:
        return (
            " + ".join(by_request[k]["label"] for k in request_keys if k in by_request)
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
                "engine_submit",
                "scheduler_queue_episode",
                "detokenize_duration_ns",
                "request_hash_call",
            },
            "shared-row": {
                "scheduler_schedule",
                "request_hash_call",
                "manager_ranges_ns.lookup",
                "external_lookup",
                "manager_ranges_ns.alloc",
                "other_ns",
                "engine_scheduler_update",
            },
            "worker-row": {
                key for key in keys if key.startswith(("worker.", "runner.", "kv."))
            },
            "task-row": set(values),
            "fence-row": {"pool_save_fence"},
            "native-row": {
                key for key in keys if key.startswith("phase") or key == "native_total"
            },
        }.get(kind, set(values))
        empty = {
            key: (
                '<span class="missing">未采集</span>'
                if key in applicable
                else '<span class="missing" title="不属于此行统计范围">—</span>'
            )
            for key in keys
        }
        return (
            '<tr class="'
            + kind
            + '" data-requests="'
            + escape(" ".join(request_keys))
            + '" title="'
            + escape(title)
            + '">'
            + '<td class="request">'
            + escape(
                identity_label if identity_label is not None else labels(request_keys)
            )
            + "</td>"
            + '<td class="step">'
            + escape(step)
            + "</td>"
            + '<td class="tp">'
            + escape(tp)
            + "</td>"
            + "".join(
                (
                    '<td rowspan="'
                    + str(shared_span)
                    + '" data-stage="'
                    + escape(key)
                    + '">'
                    + shared_cells[key]
                    + "</td>"
                )
                if kind == "worker-row" and key in shared_cells and shared_span
                else ""
                if kind == "worker-row" and key in shared_cells
                else '<td data-stage="'
                + escape(key)
                + '">'
                + values.get(key, empty[key])
                + "</td>"
                for key in keys
            )
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
                + escape(path[depth])
                + "</th>"
            )
            index = end
    pinned = (
        f'<th class="request" rowspan="{header_depth}">请求</th>'
        f'<th class="step" rowspan="{header_depth}">step</th>'
        f'<th class="tp" rowspan="{header_depth}">TP</th>'
    )
    head = "<tr>" + pinned + "".join(headers[0])
    head += f'<th rowspan="{header_depth}">tokens / 状态 / 任务</th></tr>'
    head += "".join("<tr>" + "".join(h) + "</tr>" for h in headers[1:])
    body = []
    for request in dashboard["requests"]:
        values = {}
        for stage in request["overview"]:
            if stage["event"] in {
                "api_resolve",
                "render",
                "tokenize",
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
        if request["hash_ranges"]:
            values["request_hash_call"] = "<br>".join(
                duration(h["measurement"])
                + ("（区间不完整）" if h["complete"] is not True else "")
                for h in request["hash_ranges"]
            )
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
    native_task_keys = {
        link["task_key"]
        for link in dashboard.get("native_links", [])
        if link.get("task_key")
    }

    def compact_task(task: dict) -> bool:
        return (
            task["idle"]
            and task["key"] not in native_task_keys
            and not task["stages"]
            and not task["counters"]
            and bool(task["statuses"])
            and set(task["statuses"])
            <= {"no_task", "missing_load_spec", "nothing_to_do"}
        )

    def task_row(task: dict) -> str:
        shown_tasks.add(task["key"])
        values = measures(task["stages"])
        values["task_status"] = escape(" / ".join(task["statuses"]))
        submitted_keys = []
        submitted_mib = []
        result_counts = []
        for counters in task["counters"]:
            # Outer put and nested Mooncake counters describe the same work.
            # Range submissions and commit metadata are different semantics.
            operation = next(
                (
                    name
                    for name in (
                        "put",
                        "batch_put_from_multi_buffer_ranges",
                        "mooncake_put",
                    )
                    if name + "_submitted_keys" in counters
                    or name + "_submitted_bytes" in counters
                ),
                None,
            )
            if operation is None:
                continue
            prefix = escape(operation) + ": "
            submitted_keys.append(
                prefix + escape(counters.get(operation + "_submitted_keys"))
            )
            payload = counters.get(operation + "_submitted_bytes")
            submitted_mib.append(
                prefix + (f"{payload / 1048576:,.6f}" if _number(payload) else "—")
            )
            for suffix, label in (
                ("succeeded_keys", "成功"),
                ("failed_keys", "失败"),
                ("unknown_result_keys", "未知"),
                ("unreported_keys", "未报告"),
            ):
                value = counters.get(operation + "_" + suffix)
                if value is not None:
                    result_counts.append(label + ": " + escape(value))
        values["task_keys"] = "<br>".join(submitted_keys) or "—"
        values["task_mib"] = "<br>".join(submitted_mib) or "—"
        values["task_counters"] = "<br>".join(result_counts) or "—"
        native_values: dict[str, list[str]] = defaultdict(list)
        associations = []
        for link in dashboard.get("native_links", []):
            if link.get("task_key") != task["key"]:
                continue
            native = link["native"]
            association = link.get("call_association") or link.get(
                "association", "exact task"
            )
            associations.append(str(association))
            for field, value in native["duration_fields_ns"].items():
                field = "native_total" if field == "total" else field
                rendered = duration({"state": "measured", "duration_ns": value})
                if (
                    field == "native_total"
                    and native["timing_scope"] == "prefer_same_node_remaining"
                ):
                    rendered = "phase2–4: " + rendered
                native_values[field].append(
                    '<span title="'
                    + escape(
                        str(native.get("log_source"))
                        + "; "
                        + str(association)
                        + "; timing_scope="
                        + native["timing_scope"]
                        + "; call_record_index="
                        + str(link.get("call_record_index"))
                    )
                    + '">'
                    + rendered
                    + "</span>"
                )
        values.update(
            {field: "<br>".join(items) for field, items in native_values.items()}
        )
        if "pd_transfer_task_finished" in values:
            if "queue_wait" in values:
                values["pd_queue_wait"] = values.pop("queue_wait")
        else:
            values["task_total"] = "<br>".join(
                duration(e) for e in task["envelopes"] if e["event"] == "pool_task"
            ) or duration({})
        ranks = list(
            dict.fromkeys(
                str(r["tp_rank"])
                for r in evidence
                if str(r.get("task_id")) == task["task_id"]
                and list(clock(r)) == task["domain"]
                and r.get("tp_rank") is not None
            )
        )
        steps = [step_numbers[e] for e in task["execution_ids"] if e in step_numbers]
        note = (
            "/".join(task["operation"])
            + " · task="
            + task["task_id"]
            + " · execution="
            + ",".join(task["execution_ids"])
            + " · "
            + "/".join(task["statuses"])
            + (" · native: " + "; ".join(associations) if associations else "")
        )
        return row(
            task["request_keys"],
            ",".join(steps) or "未关联",
            ",".join(ranks) if ranks else "—",
            values,
            note,
            "task-row",
            str(task["domain"]) + "; " + str(task["counters"]),
        )

    for execution in dashboard["executions"]:
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
        workers = sorted(
            execution["workers"],
            key=lambda w: (
                str(w["domain"][3:5]),
                w["domain"][-1] if isinstance(w["domain"][-1], int) else -1,
                str(w["domain"][:3]),
            ),
        )
        shared_span = len(workers)
        for worker in workers:
            domain = worker["domain"]
            values = measures(worker["stages"])
            for phase, coverage in worker.get("coverage", {}).items():
                for field in ("duration_ns", "covered_ns", "uncovered_ns"):
                    value = coverage.get(field)
                    values[f"coverage.{phase}.{field}"] = (
                        '<span title="'
                        + escape(coverage.get("state"))
                        + '">'
                        + duration({"state": "measured", "duration_ns": value})
                        + "</span>"
                    )
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
        for task in dashboard["tasks"]:
            if (
                task["key"] not in shown_tasks
                and execution["execution_id"] in task["execution_ids"]
                and not compact_task(task)
            ):
                body.append(task_row(task))
    compact_groups: dict[tuple, list[dict]] = defaultdict(list)
    for task in dashboard["tasks"]:
        if task["key"] in shown_tasks:
            continue
        if compact_task(task):
            compact_groups[
                (
                    tuple(task["execution_ids"]),
                    tuple(task["operation"]),
                    tuple(task["statuses"]),
                    tuple(task["request_keys"]),
                )
            ].append(task)
        else:
            body.append(task_row(task))
    for (
        execution_ids,
        operations,
        statuses,
        request_keys,
    ), tasks in compact_groups.items():
        ranks = sorted(
            {
                str(record["tp_rank"])
                for task in tasks
                for record in evidence
                if str(record.get("task_id")) == task["task_id"]
                and list(clock(record)) == task["domain"]
                and record.get("tp_rank") is not None
            }
        )
        body.append(
            row(
                list(request_keys),
                ",".join(step_numbers[e] for e in execution_ids if e in step_numbers)
                or "未关联",
                ",".join(ranks) or "—",
                {},
                "/".join(operations)
                + f" × {len(tasks)} 个空任务 · "
                + "/".join(statuses),
                "idle-row",
                "; ".join(task["key"] for task in tasks),
            )
        )
        shown_tasks.update(task["key"] for task in tasks)
    for record in evidence:
        if record["event"] == "pool_save_fence":
            body.append(
                row(
                    [
                        r["key"]
                        for r in dashboard["requests"]
                        if any(
                            a["kind"] == "request_id"
                            and a["value"] in record.get("request_ids", [])
                            and a.get("domain") == list(clock(record))
                            for a in r["aliases"]
                        )
                    ],
                    "释放前",
                    str(record.get("tp_rank", "?")),
                    {"pool_save_fence": duration(_measurement([record]))},
                    "释放请求的保存 fence；不计入后续 step",
                    "fence-row",
                    str(clock(record)),
                )
            )
    pending_native = [
        link
        for link in dashboard.get("native_links", [])
        if not link.get("task_key") or link["task_key"] not in shown_tasks
    ]

    out = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>TTFT · step × TP 业务阶段表</title><style>"
        "body{font:14px/1.5 system-ui,sans-serif;margin:20px;color:#213047;background:#fff}"
        "h1{font-size:22px;margin:0 0 8px}p{margin:8px 0;color:#536174}"
        "select{padding:6px;max-width:80%}.sheet{overflow:auto;max-height:80vh;margin-top:12px}"
        "table{border-collapse:separate;border-spacing:0;font-size:13px;white-space:nowrap}"
        "th,td{border-right:1px solid #d5dfeb;border-bottom:1px solid #d5dfeb;"
        "padding:7px 6px;text-align:right;min-width:54px}"
        "thead{position:sticky;top:0;z-index:4}th{text-align:center;color:#182635}"
        "th.l1{background:#edf4ff}th.l2{background:#eef7f1}th.l3{background:#fff4df}"
        ".request,.step,.tp{position:sticky;z-index:2;background:inherit;text-align:left}"
        ".request{left:0;min-width:120px;max-width:120px;overflow:hidden;text-overflow:ellipsis}"
        ".step{left:133px;min-width:50px;max-width:50px;text-align:center}"
        ".tp{left:196px;min-width:36px;max-width:36px;text-align:center;font-weight:bold}"
        "thead .request,thead .step,thead .tp{background:#f5f8fa;z-index:5}"
        ".request-row{background:#eaf1fa;font-weight:600}.shared-row{background:#eff4f8}"
        ".worker-row{background:#fff}.task-row{background:#eff8f1}"
        ".note{text-align:left;max-width:350px;white-space:normal;min-width:200px}"
        ".missing{color:#b1bac5}details{margin-top:16px}summary{cursor:pointer}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:60vh;overflow:auto}"
        "[hidden]{display:none!important}</style></head><body>"
        "<h1>TTFT · step × TP 业务阶段表</h1>"
        "<p>单位 ms · 按业务阶段横向阅读；Scheduler 在同一 execution 的 TP 行间共享。"
        "请求总计与 TP 实际工作分行，左侧三列固定。</p>"
        '<label for="request-select">请求：</label><select id="request-select">'
        '<option value="all">全部请求</option>',
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
        "</select><p>请求级行看 API、TTFT 及 sampling 之后的 Frontend detokenizer；"
        "数字 TP 行看 Worker；任务子行保留原始 task / execution / TP 身份。"
        "— 是本行不适用；未采集是没有有效计时；未关联是缺少身份关联。"
        "wait_for_save 下方为保存相关明细，不代表相加等于等待接口时间；"
        "异步任务外框、put 子阶段、并行 TP 均不相加。计数原样展示，"
        "keys 是标注 API 的提交计数（range 不等于完整 KV key），MiB 是提交载荷，"
        "不叠加外层与 Mooncake 内层计数，也不是网络实测。"
        "coverage 仅取当前 Worker 的派生证据；列顺序是业务分组，不断言所有分支的调用先后。</p>"
    )
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
    if report.get("parse_errors"):
        out.append("<p>部分日志解析失败，详见末尾证据。</p>")
    if not dashboard["requests"]:
        out.append("<p>没有可关联的请求。</p>")
    out.append(
        '<div class="sheet"><table><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )
    if pending_native:
        out.append(
            '<details id="pending-native"><summary>Mooncake 待关联证据（'
            + str(len(pending_native))
            + " 条；不计入请求 / TP）</summary>"
            "<p>线程归属不等于保存任务身份；仅精确 task_key 关联的阶段进入原保存任务 put 单元格。</p><pre>"
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
