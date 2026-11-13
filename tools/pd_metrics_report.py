#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read schema-2 compact P/D logs using only the Python standard library.

Usage: python tools/pd_metrics_report.py service.log -o report.json [--html]
Durations are evidence, not additive TTFT attribution. Native logs are optional.
"""

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MARKER = "[PD_DIAG] "
EVENTS = {
    "api_summary",
    "core_summary",
    "worker_summary",
    "pool_task",
    "pd_transfer_summary",
}
IDENTITY = (
    "host",
    "pid",
    "run_id",
    "native_tid",
    "request_id",
    "observation_id",
    "request_generation",
    "engine_id",
    "execution_id",
    "task_id",
    "dp_rank",
    "pp_rank",
    "tp_rank",
    "phase",
    "operation",
)
NATIVE_PHASES = {
    "phase1_start_batch_put": "start_batch_put",
    "phase2_submit_transfers": "submit_transfers",
    "phase3_wait_transfers": "wait_transfers",
    "phase4_finalize_batch_put": "finalize_batch_put",
}
NATIVE_SCOPES = {"normal_full", "prefer_same_node_full", "prefer_same_node_not_started"}
_GLOG = re.compile(
    r"\b[IWEF](\d{8}|\d{4})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{1,9})\s+(\d+)\s+\S+:\d+\]"
)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def clock(record: dict) -> tuple:
    return tuple(record.get(k) for k in ("host", "pid", "run_id"))


def _key(*values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode()
    ).hexdigest()[:20]


def _side(record: dict) -> str | None:
    roles = {
        "P": "P",
        "p": "P",
        "prefill": "P",
        "producer": "P",
        "kv_producer": "P",
        "D": "D",
        "d": "D",
        "decode": "D",
        "consumer": "D",
        "kv_consumer": "D",
    }
    found = {
        roles[record[k]]
        for k in ("side", "kv_role", "role", "transfer_role")
        if isinstance(record.get(k), str) and record[k] in roles
    }
    if len(found) == 1:
        return found.pop()
    if (
        not found
        and record.get("event") == "pd_transfer_summary"
        and record.get("direction") == "read"
    ):
        return "D"
    return None


def _validate(record: Any) -> None:
    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record["version"] != 2
    ):
        raise ValueError(
            "schema 2 required; schema 1/mixed-version logs are unsupported"
        )
    if record.get("event") not in EVENTS:
        raise ValueError(
            "unsupported schema-2 summary event: " + str(record.get("event"))
        )
    for name in ("host", "run_id", "record_id"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError("missing/invalid " + name)
    for name in ("pid", "native_tid", "timestamp_ns"):
        if (
            not isinstance(record.get(name), int)
            or isinstance(record[name], bool)
            or record[name] < 0
        ):
            raise ValueError("missing/invalid " + name)
    stages = record.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be an object")
    for name, stage in stages.items():
        if not isinstance(name, str) or not isinstance(stage, dict):
            raise ValueError("invalid stage object")
        if (
            not isinstance(stage.get("calls"), int)
            or isinstance(stage["calls"], bool)
            or stage["calls"] < 0
        ):
            raise ValueError("stage " + name + " requires nonnegative integer calls")
        if not isinstance(stage.get("status"), str):
            raise ValueError("stage " + name + " requires status")
        for field in ("duration_ns", "start_ns", "end_ns"):
            if stage.get(field) is not None and not _number(stage[field]):
                raise ValueError("invalid stage " + name + "." + field)
    for name in ("members", "links", "native_calls"):
        if name in record and (
            not isinstance(record[name], list)
            or any(not isinstance(item, dict) for item in record[name])
        ):
            raise ValueError(name + " must be an array of objects")
    if "official" in record and not isinstance(record["official"], dict):
        raise ValueError("official must be an object")


def _native_timing(line: str) -> dict | None:
    marker = next(
        (m for m in ("[BatchPutTiming]", "[BatchUpsertTiming]") if m in line), None
    )
    if marker is None:
        return None
    prefix, body = line.split(marker, 1)
    fields = dict(re.findall(r"(\w+)=([^\s]+)", body))
    durations = {}
    for name in (*NATIVE_PHASES, "total"):
        field = name + "_us"
        if field in fields:
            value = float(fields[field])
            if not _number(value):
                raise ValueError("invalid native duration: " + field)
            durations[name] = value * 1000
    scope = fields.get("timing_scope")
    if scope not in NATIVE_SCOPES:
        raise ValueError("unsupported native timing_scope: " + str(scope))
    if fields.get("status") not in ("ok", "error") or fields.get("complete") not in (
        "true",
        "false",
    ):
        raise ValueError("native status and complete evidence required")
    if fields["complete"] == "true" and set(durations) != {*NATIVE_PHASES, "total"}:
        raise ValueError("complete native record requires four phases and outer total")
    not_started = scope == "prefer_same_node_not_started"
    if not durations and not not_started and fields.get("complete") != "false":
        raise ValueError("complete native timing record has no retained durations")
    if not_started and (
        fields.get("status") != "error"
        or fields.get("complete") != "false"
        or fields.get("phase_status") != "not_started"
    ):
        raise ValueError("invalid native not-started evidence")
    allowed = set() if not_started else {*NATIVE_PHASES, "total"}
    if not set(durations) <= allowed:
        raise ValueError("native phases conflict with timing_scope " + scope)
    glog = _GLOG.search(prefix)
    thread = re.search(r"\b(?:native_tid|native_thread_id|tid)=(\d+)", prefix)
    pid = re.search(r"\bpid=(\d+)", prefix)
    host = re.search(r"\bhost=([^\s]+)", prefix)
    run = re.search(r"\brun_id=([^\s]+)", prefix)
    return {
        "event": "mooncake_native_timing",
        "native_kind": marker,
        "native_fields": fields,
        "duration_fields_ns": durations,
        "timing_scope": scope,
        "host": host[1] if host else None,
        "run_id": run[1] if run else None,
        "pid": int(pid[1]) if pid else None,
        "native_tid": int(thread[1]) if thread else int(glog[6]) if glog else None,
        "glog_wall": list(glog.groups()[:5]) if glog else None,
    }


def read_records(paths: list[Path]) -> tuple[list[dict], list[str]]:
    records, errors, seen = [], [], {}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                if MARKER not in line and not any(
                    m in line for m in ("[BatchPutTiming]", "[BatchUpsertTiming]")
                ):
                    continue
                try:
                    if MARKER in line:
                        record = json.loads(line.split(MARKER, 1)[1])
                        _validate(record)
                        identity = (*clock(record), record["record_id"])
                        if identity in seen:
                            if seen[identity] != record:
                                raise ValueError("conflicting duplicate record_id")
                            continue
                        seen[identity] = dict(record)
                    else:
                        record = _native_timing(line)
                    record["log_source"] = {"file": str(path), "line": line_no}
                    records.append(record)
                except (ValueError, TypeError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
    return records, errors


def _measurement(stage: dict | None) -> dict:
    if stage is None:
        return {
            "state": "missing",
            "duration_ns": None,
            "duration_ms": None,
            "status": "missing",
            "calls": 0,
            "timing_scope": None,
        }
    status = stage.get("status", "missing")
    duration = stage.get("duration_ns")
    start, end = stage.get("start_ns"), stage.get("end_ns")
    invalid = (start is not None and end is not None and end < start) or (
        duration is not None and not _number(duration)
    )
    absent = {
        "missing",
        "skipped",
        "fused",
        "not_measured",
        "unmeasured",
        "invalid",
        "not_started",
    }
    state = (
        "invalid"
        if invalid or status == "invalid"
        else status
        if status in absent
        else "measured"
        if _number(duration)
        else "missing"
    )
    duration = duration if state == "measured" else None
    return {
        **stage,
        "state": state,
        "duration_ns": duration,
        "duration_ms": duration / 1_000_000 if duration is not None else None,
        "timing_scope": stage.get(
            "timing_scope", "cumulative" if stage.get("calls", 0) > 1 else "interval"
        ),
    }


def _members(record: dict) -> list[dict]:
    if record["event"] == "api_summary" and record.get("scope") != "request":
        return [
            {
                "request_id": None,
                "observation_id": record.get("observation_id"),
                "submission_id": record.get("submission_id"),
            }
        ]
    result = []
    if record.get("request_id") is not None:
        result.append(
            {
                name: record[name]
                for name in (
                    "request_id",
                    "observation_id",
                    "request_generation",
                    "engine_id",
                )
                if record.get(name) is not None
            }
        )
    for member in record.get("members", []):
        if member.get("request_id") is not None and member not in result:
            result.append(member)
    for request_id in record.get("request_ids", []):
        if not any(m.get("request_id") == request_id for m in result):
            result.append({"request_id": request_id})
    return result


def _wall_inside(native: dict, call: dict) -> bool:
    wall = native.get("glog_wall")
    start, end = call.get("start_wall_ns"), call.get("end_wall_ns")
    offset = call.get("wall_utc_offset_seconds")
    if (
        not wall
        or not _number(start)
        or not _number(end)
        or end < start
        or not isinstance(offset, (int, float))
        or isinstance(offset, bool)
        or not math.isfinite(offset)
        or abs(offset) > 86400
    ):
        return False
    date, hour, minute, second, fraction = wall
    try:
        local_years = {
            datetime.fromtimestamp(value / 1e9 + offset, UTC).year
            for value in (start, end)
        }
    except (ValueError, OverflowError, OSError):
        return False
    years = {int(date[:4])} if len(date) == 8 else local_years
    monthday = date[-4:]
    for year in years:
        try:
            base = datetime(
                year,
                int(monthday[:2]),
                int(monthday[2:]),
                int(hour),
                int(minute),
                int(second),
                tzinfo=UTC,
            )
            stamp = (
                int(base.timestamp()) * 1_000_000_000
                + int(fraction.ljust(9, "0"))
                - int(offset * 1e9)
            )
        except (ValueError, OverflowError, OSError):
            continue
        # The printed precision denotes a bucket; overlapping calls are ambiguous.
        if stamp <= end and stamp + 10 ** (9 - len(fraction)) - 1 >= start:
            return True
    return False


def _native_links(records: list[dict], quality: list[dict]) -> dict[int, list[dict]]:
    calls = []
    for index, record in enumerate(records):
        if record["event"] != "pool_task":
            continue
        for number, call in enumerate(record.get("native_calls", [])):
            calls.append((index, number, call, record))
    links = defaultdict(list)
    assigned = defaultdict(list)
    for native_index, native in enumerate(records):
        if native["event"] != "mooncake_native_timing":
            continue
        if native["native_fields"].get("complete") == "false":
            quality.append(
                {
                    "kind": "native_incomplete",
                    "record_index": native_index,
                    "source_observations": [native],
                }
            )
        candidates = []
        for index, number, call, record in calls:
            domain = {**record, **call}
            if native.get("native_tid") is None or native["native_tid"] != call.get(
                "native_tid"
            ):
                continue
            if any(
                native.get(k) is not None and native[k] != domain.get(k)
                for k in ("host", "pid", "run_id")
            ):
                continue
            source = record.get("log_source", {}).get("file")
            same_source = source is not None and source == native.get(
                "log_source", {}
            ).get("file")
            if not same_source and not all(
                native.get(k) is not None for k in ("host", "pid", "run_id")
            ):
                continue
            if not _wall_inside(native, call):
                continue
            fields = native["native_fields"]
            # Counts only disqualify an otherwise identified time/thread sample.
            if any(
                str(call[k]) != fields[n]
                for k, n in (("submitted_keys", "keys"), ("submitted_bytes", "bytes"))
                if call.get(k) is not None and n in fields
            ):
                continue
            candidates.append((index, number, call))
        if len(candidates) != 1:
            quality.append(
                {
                    "kind": "native_unmatched"
                    if not candidates
                    else "native_ambiguous",
                    "record_index": native_index,
                    "candidate_count": len(candidates),
                    "source_observations": [native],
                }
            )
            continue
        index, number, call = candidates[0]
        assigned[(index, number)].append((native_index, native))
    for (index, number), natives in assigned.items():
        valid = len(natives) == 1
        if not valid:
            quality.append(
                {
                    "kind": "native_ambiguous",
                    "record_index": index,
                    "call_index": number,
                    "reason": "multiple native samples for one call",
                }
            )
            continue
        links[index].append(
            {
                "call_index": number,
                "call": records[index]["native_calls"][number],
                "natives": [{"record_index": ni, **native} for ni, native in natives],
            }
        )
    return links


def build_dashboard(records: list[dict]) -> dict:
    for record in records:
        if record.get("event") != "mooncake_native_timing":
            _validate(record)
    quality, nodes, requests, observations, tasks, transfers = [], [], [], [], [], []
    summaries = [(i, r) for i, r in enumerate(records) if r["event"] in EVENTS]
    record_members = {i: _members(r) for i, r in summaries}
    execution_members = defaultdict(list)
    for _, record in summaries:
        if record["event"] == "core_summary" and record.get("scope") == "execution":
            execution_members[record.get("execution_id")].append(record)
    for index, record in summaries:
        if record["event"] == "worker_summary":
            candidates = execution_members.get(record.get("execution_id"), [])
            if len(candidates) == 1:
                core = candidates[0]
                record_members[index] = [
                    {
                        **{
                            k: core[k]
                            for k in (
                                "engine_id",
                                "observation_id",
                                "request_generation",
                            )
                            if core.get(k) is not None
                        },
                        **member,
                    }
                    for member in _members(core)
                ]
            else:
                record_members[index] = []
                quality.append(
                    {
                        "kind": "unassigned_worker_execution",
                        "record_index": index,
                        "candidate_count": len(candidates),
                    }
                )
        elif record["event"] == "pool_task":
            enriched = []
            for member in record_members[index]:
                execution_id = member.get("execution_id", record.get("execution_id"))
                cores = execution_members.get(execution_id, [])
                matches = [
                    {
                        **{
                            k: core[k]
                            for k in ("engine_id", "observation_id")
                            if core.get(k) is not None
                        },
                        **candidate,
                    }
                    for core in cores
                    for candidate in _members(core)
                    if candidate.get("request_id") == member.get("request_id")
                ]
                enriched.append(
                    {**matches[0], **member} if len(matches) == 1 else member
                )
            record_members[index] = enriched
    evidence = [
        {
            **{
                k: r[k]
                for k in ("engine_id", "observation_id", "request_generation")
                if r.get(k) is not None
            },
            **m,
        }
        for i, r in summaries
        for m in record_members[i]
        if m.get("request_id") is not None
    ]
    anchors = {}

    def resolve(member, record):
        rid = member.get("request_id")
        fields = {
            k: member.get(k, record.get(k))
            for k in ("engine_id", "observation_id", "request_generation")
        }
        if rid is None:
            ident = (
                "observation",
                *clock(record),
                record.get("scope"),
                fields["observation_id"],
                member.get("submission_id"),
                record["record_id"],
            )
        else:
            # Request IDs alone cannot bridge processes or reused lifecycles.
            candidates = [
                m
                for m in evidence
                if m.get("request_id") == rid
                and any(
                    fields[k] is not None and m.get(k) == fields[k]
                    for k in ("engine_id", "observation_id")
                )
                and all(
                    value is None or m.get(k) in (None, value)
                    for k, value in fields.items()
                )
            ]
            for field in fields:
                values = {m[field] for m in candidates if m.get(field) is not None}
                if fields[field] is None and len(values) == 1:
                    fields[field] = next(iter(values))
            domain = (
                None
                if fields["engine_id"] is not None
                or fields["observation_id"] is not None
                else clock(record)
            )
            lifecycle = fields["observation_id"]
            generation = fields["request_generation"] if lifecycle is None else None
            ident = ("request", rid, fields["engine_id"], lifecycle, generation, domain)
        anchors.setdefault(
            ident,
            {
                "request_id": rid,
                **fields,
                "observation_only": rid is None,
                "side": _side(record),
            },
        )
        if anchors[ident]["side"] is None and _side(record) is not None:
            anchors[ident]["side"] = _side(record)
        return ident

    membership = {
        i: list(
            dict.fromkeys(
                resolve(m, r) for m in (record_members[i] or [{"request_id": None}])
            )
        )
        for i, r in summaries
    }
    transport_links = []
    parents = {ident: ident for ident in anchors}

    def root(ident):
        while parents[ident] != ident:
            ident = parents[ident]
        return ident

    def endpoint(engine, rid):
        matches = [
            ident
            for ident, fact in anchors.items()
            if engine is not None
            and rid is not None
            and fact["engine_id"] == engine
            and fact["request_id"] == rid
        ]
        return matches[0] if len(matches) == 1 else None

    for index, record in summaries:
        links = list(record.get("links", []))
        if (
            record.get("remote_request_id") is not None
            and record.get("remote_engine_id") is not None
        ):
            links.append(
                {
                    "kind": "producer_consumer",
                    "producer_request_id": record["remote_request_id"],
                    "producer_engine_id": record["remote_engine_id"],
                    "consumer_request_id": record.get("request_id"),
                    "consumer_engine_id": record.get("engine_id"),
                }
            )
        for link in links:
            transport_links.append(
                {"record_index": index, "source_observations": [record], **link}
            )
            if link.get("kind") not in ("producer_consumer", "producer"):
                continue
            local = endpoint(
                link.get("consumer_engine_id", record.get("engine_id")),
                link.get("consumer_request_id", record.get("request_id")),
            )
            remote = endpoint(
                link.get("producer_engine_id", link.get("engine_id")),
                link.get("producer_request_id", link.get("request_id")),
            )
            if local is not None and remote is not None:
                anchors[local]["side"], anchors[remote]["side"] = "D", "P"
                if root(local) != root(remote):
                    parents[root(local)] = root(remote)
            else:
                quality.append(
                    {
                        "kind": "unresolved_transport_link",
                        "record_index": index,
                        "source_observations": [record],
                        "link": link,
                    }
                )
    native_links = _native_links(records, quality)
    by_request = defaultdict(list)
    for index, identities in membership.items():
        for ident in identities:
            if index not in by_request[root(ident)]:
                by_request[root(ident)].append(index)
    request_keys = {ident: "request-" + _key(*ident) for ident in by_request}
    for link in transport_links:
        if link.get("kind") not in ("producer_consumer", "producer"):
            continue
        record = records[link["record_index"]]
        local = endpoint(
            link.get("consumer_engine_id", record.get("engine_id")),
            link.get("consumer_request_id", record.get("request_id")),
        )
        remote = endpoint(
            link.get("producer_engine_id", link.get("engine_id")),
            link.get("producer_request_id", link.get("request_id")),
        )
        if local is not None and remote is not None:
            link["source_request_key"] = request_keys[root(remote)]
            link["target_request_key"] = request_keys[root(local)]
    for index, record in summaries:
        if record["event"] != "api_summary" or record.get("scope") == "request":
            continue
        target_ids = {
            link.get("request_id")
            for link in record.get("links", [])
            if link.get("kind") in ("request_id", "engine_request", "request_child")
        }
        for ident, fact in anchors.items():
            if fact["request_id"] is None:
                continue
            same_observation = (
                record.get("observation_id") is not None
                and fact["observation_id"] == record["observation_id"]
            )
            if same_observation and (
                not target_ids or fact["request_id"] in target_ids
            ):
                transport_links.append(
                    {
                        "kind": "api_observation",
                        "record_index": index,
                        "source_request_key": request_keys[root(membership[index][0])],
                        "target_request_key": request_keys[root(ident)],
                        "source_observations": [record],
                    }
                )
    roots = {}

    def node(
        request_key,
        side,
        parent,
        key,
        label,
        record=None,
        stage=None,
        kind="stage",
        **extra,
    ):
        measurement = _measurement(stage)
        source = [] if record is None else [record]
        node_id = "n-" + _key(
            request_key,
            side,
            parent,
            key,
            record.get("record_id") if record else None,
            extra.get("sample_id"),
        )
        parent_node = next((n for n in reversed(nodes) if n["node_id"] == parent), None)
        path = (parent_node["path"] + " / " if parent_node else "") + label
        hierarchy = (parent_node["hierarchy"] if parent_node else []) + [key]
        row = {
            "node_id": node_id,
            "parent_id": parent,
            "request_key": request_key,
            "side": side,
            "level": "L" + str(int(parent_node["level"][1:]) + 1 if parent_node else 1),
            "node_key": key,
            "label": label,
            "path": path,
            "hierarchy": hierarchy,
            **{
                f"l{i + 1}": hierarchy[i] if i < len(hierarchy) else None
                for i in range(4)
            },
            "scope": record.get(
                "scope", record.get("phase", record.get("operation", "entry"))
            )
            if record
            else "request",
            "repository": "mooncake"
            if kind == "native"
            else "vllm-ascend"
            if record and record["event"] in ("pool_task", "pd_transfer_summary")
            else "vllm",
            "kind": kind,
            **measurement,
            "execution_id": record.get("execution_id") if record else None,
            "task_id": record.get("task_id") if record else None,
            "rank": {k: record.get(k) for k in ("dp_rank", "pp_rank", "tp_rank")}
            if record
            else None,
            "source_observations": source,
            "completion": record.get("completion", record.get("complete"))
            if record
            else None,
            "child_count": 0,
            **({k: record.get(k) for k in IDENTITY} if record else {}),
        }
        # Keep the public identity fields, but never overwrite canonical keys.
        row.update(
            {
                "node_id": node_id,
                "request_key": request_key,
                "side": side,
                "scope": extra.pop("scope", row.get("scope", "request")),
                "source_observations": source,
                "child_count": 0,
            }
        )
        row.update(extra)
        if stage is not None and "complete" in stage:
            row["completion"] = stage["complete"]
        if kind in ("group", "association", "event"):
            row.update(
                state=kind, duration_ns=None, duration_ms=None, timing_scope=None
            )
        nodes.append(row)
        if parent_node:
            parent_node["child_count"] += 1
        return node_id

    for ident, indices in by_request.items():
        key = "request-" + _key(*ident)
        identities = [
            anchors[candidate] for candidate in anchors if root(candidate) == ident
        ]
        facts = {**anchors[ident], "identities": identities}
        request_records = [records[i] for i in indices]

        def record_side(index, ident=ident):
            explicit = _side(records[index])
            values = {anchors[m]["side"] for m in membership[index] if root(m) == ident}
            return (
                explicit
                if explicit is not None
                else next(iter(values))
                if len(values) == 1
                else None
            )

        sides = list(dict.fromkeys(record_side(i) for i in indices))
        official = [
            {
                "side": record_side(i),
                "official": records[i]["official"],
                "source_observations": [records[i]],
            }
            for i in indices
            if records[i]["event"] == "api_summary" and "official" in records[i]
        ]
        requests.append(
            {
                "key": key,
                "request_key": key,
                **facts,
                "sides": sides,
                "official_summaries": official,
                "source_observations": request_records,
                "links": [
                    link
                    for link in transport_links
                    if link["record_index"] in indices
                    or link.get("target_request_key") == key
                ],
                "ttft_attribution": "not_established",
            }
        )
        for side in sides:
            for group, label in (
                ("api", "API Server"),
                ("core", "EngineCore"),
                ("worker", "Worker"),
            ):
                roots[(key, side, group)] = node(
                    key, side, None, group, label, kind="group"
                )
        for index in indices:
            record = records[index]
            side, stages, event = (
                record_side(index),
                record.get("stages", {}),
                record["event"],
            )
            context = " · ".join(
                f"{k}={record[k]}"
                for k in ("execution_id", "task_id", "tp_rank", "phase", "operation")
                if record.get(k) is not None
            )
            shared = len(membership[index]) > 1
            common = {
                "shared_batch": shared,
                "member_count": len(membership[index]),
                "record_index": index,
            }

            def add(
                parent,
                name,
                stage=None,
                *,
                key=key,
                side=side,
                context=context,
                record=record,
                stages=stages,
                common=common,
                **extra,
            ):
                return node(
                    key,
                    side,
                    parent,
                    name,
                    name + (" · " + context if context else ""),
                    record,
                    stages.get(name) if stage is None else stage,
                    **common,
                    **extra,
                )

            def add_ranges(parent, name, stage, **extra):
                if not stage.get("ranges"):
                    add(parent, name, stage, execution_id=None, **extra)
                    return
                for number, interval in enumerate(stage["ranges"]):
                    execution = interval.get("execution_id")
                    add(
                        parent,
                        name,
                        interval,
                        sample_id=f"{name}:{number}",
                        execution_id=execution,
                        scope="execution"
                        if execution is not None
                        else "request_interval",
                        interval_index=number,
                        output_batch_id=interval.get("output_batch_id"),
                        interval_evidence=interval,
                        timing_note="wait_ends_in_execution_not_execution_duration"
                        if name == "queue"
                        else "actual_call_interval",
                        attribution="explicit_execution"
                        if execution is not None
                        else "unassigned_interval",
                        ranges_complete=stage.get("complete", True),
                        dropped_ranges=stage.get("dropped_ranges", 0),
                        **extra,
                    )

            if event == "api_summary":
                parent = roots[(key, side, "api")]
                for name in (
                    "req_resolve",
                    "render",
                    "tokenize",
                    "add_request",
                    "detokenize",
                ):
                    if name in stages:
                        if name == "detokenize":
                            add_ranges(
                                parent,
                                name,
                                stages[name],
                                supplement=True,
                                endpoint="outside_official_ttft",
                            )
                        else:
                            add(parent, name)
                for link in transport_links:
                    if (
                        link["record_index"] == index
                        and link.get("kind") == "api_observation"
                    ):
                        add(
                            parent,
                            "request_link",
                            kind="association",
                            sample_id=link["target_request_key"],
                            target_request_key=link["target_request_key"],
                        )
            elif event == "core_summary":
                parent = roots[(key, side, "core")]
                if record.get("scope") == "request":
                    for name in ("queue", "hash"):
                        if name in stages:
                            add_ranges(
                                parent, name, stages[name], relationship="associated"
                            )
                else:
                    schedule = add(parent, "schedule")
                    for name in ("lookup", "alloc", "other"):
                        add(schedule, name, relationship="component")
            elif event == "worker_summary":
                parent = roots[(key, side, "worker")]
                if "sampling" in stages:
                    add(parent, "sampling")
                if "submit" in stages or "wait_for_transfer" in stages:
                    load = add(parent, "load_kv", kind="group")
                    if "submit" in stages:
                        add(
                            load,
                            "submit",
                            timing_note="submission_return_not_transfer_completion",
                        )
                    if "wait_for_transfer" in stages:
                        add(load, "wait_for_transfer")
                for name in ("forward", "wait_for_save"):
                    if name in stages:
                        add(parent, name)
            elif event == "pool_task":
                parent = roots[(key, side, "worker")]
                operation = str(record.get("operation", "task"))
                parent = add(
                    parent,
                    "load_kv" if operation == "load" else "wait_for_save",
                    kind="group",
                    scope=operation,
                )
                stage_nodes = {}
                for name in ("build_key", "lookup", "prepare", "sync", "put"):
                    if name in stages and stages[name].get("calls", 0) > 0:
                        stage_nodes[name] = add(parent, name, scope=operation)
                put = stage_nodes.get("put")
                for link in native_links.get(index, []):
                    call = link["call"]
                    number = link["call_index"]
                    if put is None:
                        put = add(parent, "put", kind="group", scope=operation)
                    for native in link["natives"]:
                        durations = native["duration_fields_ns"]
                        scope = native["timing_scope"]
                        native_source = {
                            **record,
                            "native": native,
                            "timing_scope": scope,
                        }
                        for field, name in NATIVE_PHASES.items():
                            node(
                                key,
                                side,
                                put,
                                name,
                                name,
                                native_source,
                                {
                                    "calls": 1 if field in durations else 0,
                                    "status": native["native_fields"].get(
                                        "phase_status", "missing"
                                    )
                                    if field not in durations
                                    else native["native_fields"].get("status", "ok"),
                                    "duration_ns": durations.get(field),
                                    "timing_scope": scope,
                                    "complete": native["native_fields"].get("complete")
                                    == "true",
                                },
                                kind="native",
                                sample_id=f"{number}:{scope}:{name}",
                                call_index=number,
                                call_id=call.get("call_id"),
                                native_call=call,
                                native_total={
                                    "duration_ns": durations.get("total"),
                                    "timing_scope": scope,
                                    "status": native["native_fields"].get("status"),
                                    "complete": native["native_fields"].get("complete")
                                    == "true",
                                },
                                source_observations=[record, native],
                            )
                tasks.append(
                    {
                        **record,
                        "request_key": key,
                        "side": side,
                        "record_index": index,
                        "shared_batch": shared,
                        "source_observations": [record],
                    }
                )
            elif event == "pd_transfer_summary":
                transfers.append(
                    {
                        **record,
                        "request_key": key,
                        "side": side,
                        "record_index": index,
                        "source_observations": [record],
                    }
                )
                if record.get("scope") == "task":
                    parent = add(
                        roots[(key, side, "worker")],
                        "load_kv",
                        kind="group",
                        scope="receive_task",
                    )
                    if "wait_for_transfer" in stages:
                        add(parent, "wait_for_transfer", scope="receive_task")
            fields = [
                (group, name, value)
                for group in ("stages", "associated_stages", "counters", "official")
                for name, value in record.get(group, {}).items()
            ]
            fields.extend(
                (group, str(number), value)
                for group in ("native_calls", "completions")
                for number, value in enumerate(record.get(group, []))
            )
            for link in native_links.get(index, []):
                for native in link["natives"]:
                    fields.extend(
                        ("native", f"{link['call_index']}.{name}", value)
                        for name, value in native["native_fields"].items()
                    )
            for group, name, value in fields:
                timing = group in ("stages", "associated_stages")
                unit = (
                    "ns"
                    if timing or name.endswith("_ns")
                    else "us"
                    if group == "native" and name.endswith("_us")
                    else "bytes"
                    if "bytes" in name
                    else "keys"
                    if "keys" in name
                    else "tokens"
                    if name
                    in (
                        "num_prompt_tokens",
                        "num_generation_tokens",
                        "num_cached_tokens",
                    )
                    else "seconds"
                    if group == "official"
                    and name
                    in (
                        "time_to_first_token",
                        "e2e_latency",
                        "queued_time",
                        "prefill_time",
                        "inference_time",
                        "decode_time",
                        "response_time_to_first_token",
                    )
                    else "seconds/token"
                    if name == "mean_time_per_output_token"
                    else "count"
                    if group == "counters" or name == "num_preemptions"
                    else None
                )
                observations.append(
                    {
                        **{k: record.get(k) for k in IDENTITY},
                        "request_key": key,
                        "side": side,
                        "event": event,
                        "group": group,
                        "field": name,
                        "stage": name,
                        "value": value,
                        "value_json": json.dumps(value, ensure_ascii=False),
                        "unit": unit,
                        "record_index": index,
                        "scope": record.get(
                            "scope", record.get("phase", record.get("operation"))
                        ),
                        **(
                            _measurement(value)
                            if timing
                            else {
                                "state": "missing" if value is None else "observed",
                                "status": record.get("status"),
                                "duration_ns": None,
                                "duration_ms": None,
                            }
                        ),
                        "source_observations": [record],
                        "shared_batch": shared,
                    }
                )
            positions = {
                name: record[name]
                for name in (
                    "first_generated",
                    "response_start",
                    "first_body",
                    "engine_first_output",
                )
                if isinstance(record.get(name), dict)
            }
            for name, value in positions.items():
                observations.append(
                    {
                        "request_key": key,
                        "side": side,
                        "stage": name,
                        "record_index": index,
                        "scope": record.get("scope"),
                        "kind": "event",
                        "state": "event",
                        "duration_ns": None,
                        "duration_ms": None,
                        "evidence": value,
                        "source_observations": [record],
                    }
                )
            if (
                record.get("complete") is False
                or record.get("incomplete")
                or any(
                    value
                    for name, value in record.items()
                    if name.endswith("_dropped") or name.startswith("dropped_")
                )
                or any(
                    stage.get("complete") is False or stage.get("dropped_ranges")
                    for stage in stages.values()
                )
            ):
                quality.append(
                    {
                        "request_key": key,
                        "side": side,
                        "kind": "incomplete",
                        "source_observations": [record],
                    }
                )
    return {
        "version": 2,
        "requests": requests,
        "nodes": nodes,
        "observations": observations,
        "tasks": tasks,
        "pd_transfers": transfers,
        "quality": quality,
        "transport_links": transport_links,
        "native_links": dict(native_links),
    }


def summarize(records: list[dict], errors: list[str]) -> dict:
    if errors:
        raise ValueError("Invalid diagnostic records:\n" + "\n".join(errors))
    if not any(record.get("event") in EVENTS for record in records):
        raise ValueError("No schema-2 diagnostic summaries found")
    return {
        "version": 2,
        "record_count": len(records),
        "event_counts": dict(Counter(r["event"] for r in records)),
        "records": records,
        "dashboard": build_dashboard(records),
        "warning": (
            "Diagnostics do not establish additive TTFT attribution. "
            "SYNTHETIC inputs are not performance measurements."
        ),
    }


def render_html(report: dict) -> str:
    if report.get("version") != 2:
        raise ValueError("schema-2 report required")
    dashboard = report["dashboard"]
    children = defaultdict(list)
    for row in dashboard["nodes"]:
        children[row["parent_id"]].append(row)

    def render(row):
        escape = lambda value: html.escape(str(value), quote=True)
        duration = (
            str(row["duration_ms"]) + " ms" if row["duration_ms"] is not None else "—"
        )
        title = f"{escape(row['label'])} · {escape(row['state'])} · {duration}"
        detail = {k: v for k, v in row.items() if k not in ("source_observations",)}
        body = (
            "<pre>"
            + escape(json.dumps(detail, ensure_ascii=False, indent=2))
            + "</pre>"
        )
        source = (
            "<details><summary>Source evidence</summary><pre>"
            + escape(
                json.dumps(row["source_observations"], ensure_ascii=False, indent=2)
            )
            + "</pre></details>"
        )
        return (
            '<details id="'
            + escape(row["node_id"])
            + '"><summary>'
            + title
            + "</summary>"
            + body
            + source
            + "".join(render(child) for child in children[row["node_id"]])
            + "</details>"
        )

    def matrix(key):
        def column_path(row):
            path = tuple(row["hierarchy"])
            return path + ("",) * (4 - len(path))

        paths = sorted(
            {
                column_path(row)
                for row in dashboard["nodes"]
                if row["request_key"] == key
                and row["kind"] not in ("group", "association", "event")
            }
        )
        samples = defaultdict(lambda: defaultdict(list))
        execution_order = {}
        for row in dashboard["nodes"]:
            if row["request_key"] != key or row["kind"] in ("group", "association"):
                continue
            source = row.get("source_observations", [])
            record = source[0] if source else {}
            event = record.get("event")
            worker = event in ("worker_summary", "pool_task", "pd_transfer_summary")
            execution = row.get("execution_id")
            if execution is not None and execution not in execution_order:
                execution_order[execution] = len(execution_order) + 1
            rank = row.get("rank") or {}
            identity = (
                row.get("side"),
                execution,
                rank.get("dp_rank"),
                rank.get("pp_rank") if worker else None,
                rank.get("tp_rank") if worker else None,
            )
            samples[identity][column_path(row)].append(row)
        if not samples:
            return ""
        escape = lambda value: html.escape(str(value), quote=True)
        header_rows = []
        for depth in range(4):
            cells = []
            if depth == 0:
                cells.append('<th rowspan="4">step</th>')
                cells.append('<th rowspan="4">观测范围</th>')
            index = 0
            while index < len(paths):
                path = paths[index]
                if len(path) <= depth:
                    index += 1
                    continue
                end = index + 1
                while end < len(paths) and paths[end][: depth + 1] == path[: depth + 1]:
                    end += 1
                span = (
                    4 - depth
                    if all(
                        len(candidate) == depth + 1 for candidate in paths[index:end]
                    )
                    else 1
                )
                display_name = {
                    "api": "API Server",
                    "core": "EngineCore",
                    "worker": "Worker",
                }.get(path[depth], path[depth])
                label = f"L{depth + 1} · {escape(display_name)}" if path[depth] else ""
                cells.append(
                    f'<th class="l{depth + 1}" colspan="{end - index}" '
                    f'rowspan="{span}">{label}</th>'
                )
                index = end
            header_rows.append("<tr>" + "".join(cells) + "</tr>")
        header = '<table class="step-tp"><thead>' + "".join(header_rows)
        header += "</thead><tbody>"
        body = []
        ordered = sorted(
            samples.items(),
            key=lambda item: (
                execution_order.get(item[0][1], 0),
                tuple(-1 if value is None else value for value in item[0][2:]),
                str(item[0][0]),
            ),
        )
        for identity, values in ordered:
            side, execution, dp, pp, tp = identity
            step = execution_order.get(execution, "request")
            worker_row = pp is not None or tp is not None
            rank_label = (
                f"Worker · TP {tp if tp is not None else '—'} · PP {pp} · DP {dp}"
                if worker_row
                else ("EngineCore · 共享" if execution else "请求级")
            )
            cells = [
                f'<tr><th title="{escape(execution)}">{step}</th>',
                f"<th>{escape(side or '—')} · {escape(rank_label)}</th>",
            ]
            for path in paths:
                content = []
                for row in values.get(path, []):
                    value = (
                        f"{row['duration_ms']:.3f}"
                        if row["duration_ms"] is not None
                        else row["state"]
                    )
                    evidence = escape(f"{row.get('task_id') or ''} {row['status']}")
                    content.append(
                        f'<a href="#{escape(row["node_id"])}" '
                        f'title="{evidence}">{escape(value)}</a>'
                    )
                cells.append("<td>" + ("<br>".join(content) or "—") + "</td>")
            body.append("".join(cells) + "</tr>")
        return (
            "<h3>step × TP 透视表 · 单位 ms</h3><p>"
            "step 为请求内 execution 顺序；悬停查看 execution_id。"
            "共享 Core 数据单列，不复制到每个 TP。"
            "同格多任务分别列出、不相加；点击数值查看源记录。"
            "queue 是在该 execution 结束的请求等待区间，不是本轮内部耗时。"
            "hash/detokenize 按逐次调用的 execution 身份展示；无身份保持请求级。"
            "— 为无观测，missing 为缺失。</p><div class=matrix>"
            + header
            + "".join(body)
            + "</tbody></table></div>"
        )

    def field_table(key):
        names = (
            "event",
            "group",
            "field",
            "value_json",
            "unit",
            "duration_ms",
            "state",
            "status",
            "scope",
            "execution_id",
            "task_id",
            "tp_rank",
        )
        rows = []
        for row in dashboard["observations"]:
            if row["request_key"] != key:
                continue
            cells = "".join(
                "<td>" + html.escape(str(row[name])) + "</td>"
                if row.get(name) is not None
                else "<td>—</td>"
                for name in names
            )
            source = html.escape(
                json.dumps(row["source_observations"], ensure_ascii=False, indent=2)
            )
            rows.append(
                "<tr>"
                + cells
                + "<td><details><summary>Source evidence</summary><pre>"
                + source
                + "</pre></details></td></tr>"
            )
        return (
            "<h3>全部已采集字段 · 保留单位、scope与原值</h3>"
            "<p>辅助计时、计数、官方统计和原生证据不增加业务层级；"
            "嵌套或并行观察不得相加。— 为无值，不是零。</p>"
            '<div class="matrix"><table class="field-evidence"><thead><tr>'
            + "".join("<th>" + name + "</th>" for name in names)
            + "<th>来源</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    sections = []
    options = []
    for request in dashboard["requests"]:
        key = request["request_key"]
        roots = [n for n in children[None] if n["request_key"] == key]
        title = request["request_id"] or (
            "Observation " + str(request.get("observation_id") or key)
        )
        options.append(
            '<option value="'
            + html.escape(key, quote=True)
            + '">'
            + html.escape(title)
            + "</option>"
        )
        links = "".join(
            '<p><a href="#'
            + html.escape(link["target_request_key"], quote=True)
            + '">Linked request</a></p>'
            for link in request["links"]
            if link.get("target_request_key")
        )
        sections.append(
            '<section id="'
            + html.escape(key, quote=True)
            + '"><h2>'
            + html.escape(title)
            + "</h2>"
            + links
            + matrix(key)
            + field_table(key)
            + "<details><summary>Official statistics and source evidence</summary><pre>"
            + html.escape(
                json.dumps(request["official_summaries"], ensure_ascii=False, indent=2)
            )
            + "</pre></details>"
            + "".join(render(n) for n in roots)
            + "</section>"
        )
    return (
        '<!doctype html><meta charset="utf-8"><title>Compact P/D evidence</title>'
        '<style>body{font:14px/1.6 "Segoe UI",sans-serif;margin:20px;'
        "background:#e8eaed;color:#263449}h1{font-size:20px}"
        "h2{font-size:16px;color:#4472c4}h3{font-size:14px;margin:8px 0}"
        "details{margin:6px 0 6px 16px}summary{cursor:pointer}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        ".matrix{max-width:100%;overflow:auto;background:white;"
        "box-shadow:0 1px 4px #0003;padding:2px;margin-bottom:12px}"
        "table{border-collapse:collapse;font-size:12px;line-height:1.5}"
        "th,td{border:1px solid #d0d0d0;padding:3px 8px;white-space:nowrap}"
        "td{text-align:right;font-variant-numeric:tabular-nums}"
        "thead tr:first-child th{background:#4472c4;color:white}"
        "thead tr:last-child th{background:#8eaadb;color:#18335d}"
        "thead tr:first-child th:nth-last-child(2){background:#a9d08e;color:#1a3a0a}"
        "thead tr:first-child th:last-child{background:#ffd966;color:#5a4500}"
        "tbody th{background:#f2f2f2;font-weight:500;font-size:11px}"
        "td a{color:inherit;text-decoration:none}td a:hover{text-decoration:underline}"
        ".l1{background:#4472c4!important;color:white!important}"
        ".l2{background:#8eaadb!important;color:#18335d!important}"
        ".l3{background:#a9d08e!important;color:#1a3a0a!important}"
        ".l4{background:#ffd966!important;color:#5a4500!important}"
        "details:target{outline:2px solid #4472c4}</style>"
        "<h1>Compact P/D evidence · schema 2</h1><p>"
        + html.escape(report["warning"])
        + '</p><label>请求筛选 <select id="request-select">'
        + "".join(options)
        + "</select></label>"
        + "".join(sections)
        + "<details><summary>全部质量证据</summary><pre>"
        + html.escape(json.dumps(dashboard["quality"], ensure_ascii=False, indent=2))
        + "</pre></details><script>"
        + 'const picker=document.getElementById("request-select");'
        + "function choose(key){const section=document.getElementById(key);"
        + 'if(!section||section.tagName!=="SECTION")return;picker.value=key;'
        + 'document.querySelectorAll("section").forEach(s=>s.hidden=s.id!==key);}'
        + "picker.onchange=()=>{choose(picker.value);"
        + 'history.replaceState(null,"","#"+picker.value);};'
        + 'window.addEventListener("hashchange",()=>{'
        + "const el=document.getElementById(location.hash.slice(1));"
        + 'const s=el?.closest("section");if(s)choose(s.id);});'
        + "choose(location.hash.slice(1)||picker.value);"
        + "</script>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()
    try:
        records, errors = read_records(args.logs)
        if not records and not errors:
            errors.append("No schema-2 diagnostic records found")
        report = summarize(records, errors)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.html:
        args.out.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    print(
        f"{len(records)} records; "
        f"{len(report['dashboard']['requests'])} requests -> {args.out}"
    )


if __name__ == "__main__":
    main()
