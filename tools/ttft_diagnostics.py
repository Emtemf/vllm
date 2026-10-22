#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline TTFT text reports; run with ``python -P tools/ttft_diagnostics.py``.

Only the Python standard library is required. No vLLM/native modules are imported,
and preflight never constructs a model, device, or Mooncake Store.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

MARKER = "[TTFT_DIAG]"
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s,]+)")
IDENTIFIERS = ("frontend_id", "request_id", "step_id", "task_id", "trace_id")
STRONG_IDS = ("frontend_id", "task_id", "trace_id")
LEGACY_FIELDS = (
    "ttft", "hash", "ext_lookup", "alloc", "overhead", "load_kv",
    "forward", "save_kv", "update", "residual",
)
FACT_FIELDS = (
    "status", "reason", "source", "local_tokens", "external_tokens",
    "scheduled_tokens", "num_scheduled_tokens", "total_num_scheduled_tokens",
    "num_requests", "num_scheduled_requests", "num_running_reqs",
    "num_waiting_reqs", "checked_keys", "existing_keys", "missing_keys",
    "lookup_errors", "submitted_keys", "submitted_bytes", "keys", "bytes",
    "result", "runner", "ttft_s", "residual_s",
)


class DiagnosticError(ValueError):
    """An unreadable or malformed diagnostic input."""


@dataclass
class Record:
    data: dict[str, Any]
    source: str
    line: int
    kind: str = "diagnostic"

    @property
    def event(self) -> str:
        return self.data["event"]

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line}"


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def parse_lines(lines: Iterable[str], source: str = "<stdin>") -> list[Record]:
    """Parse recognized records, ignoring unrelated log lines, never raw-dumping."""
    records = []
    for line_no, line in enumerate(lines, 1):
        try:
            if MARKER in line:
                data = json.loads(line.split(MARKER, 1)[1].strip(),
                                  parse_constant=_reject_constant)
                if not isinstance(data, dict) or not isinstance(
                    data.get("event"), str
                ) or not data["event"]:
                    raise ValueError("expected JSON object with nonempty event")
                if "version" in data and data["version"] != 1:
                    raise ValueError("unsupported TTFT_DIAG version")
                for name in IDENTIFIERS + ("span_id", "parent_span_id"):
                    if name in data and data[name] is not None and not isinstance(
                        data[name], str
                    ):
                        raise ValueError(f"{name} must be a string or null")
                for name in ("mono_ns", "wall_ns", "start_ns", "end_ns",
                             "duration_ns"):
                    if name in data and not _number(data[name]):
                        raise ValueError(f"{name} must be finite numeric nanoseconds")
                records.append(Record(data, source, line_no))
            elif "[BatchPutTiming]" in line:
                data = dict(KV.findall(line))
                if not data:
                    raise ValueError("BatchPutTiming has no fields")
                for name, value in list(data.items()):
                    if name.endswith(("_us", "_ns", "_ms")) or name in (
                        "keys", "bytes"
                    ):
                        data[name] = float(value)
                        if not _number(data[name]) or data[name] < 0:
                            raise ValueError(f"invalid native {name}")
                data["event"] = "native_batch_put"
                data["native_scope"] = (
                    "same-node phase1 (outside remaining total)"
                    if "[prefer_same_node]" in line and "total_us" not in data
                    else "same-node remaining phases (excludes phase1)"
                    if "[prefer_same_node]" in line else "batch put"
                )
                records.append(Record(data, source, line_no, "native"))
            elif any(marker in line for marker in (
                "[TTFT][BREAKDOWN]", "[TTFT BREAKDOWN]", "TTFT BREAKDOWN"
            )):
                fields = dict(KV.findall(line))
                data = {"event": "legacy_breakdown"}
                request_id = fields.get("req", fields.get("request_id"))
                if request_id:
                    data["request_id"] = request_id
                for name in LEGACY_FIELDS:
                    if name in fields:
                        value = float(fields[name])
                        if not _number(value):
                            raise ValueError(f"invalid legacy {name}")
                        data[name + "_s"] = value
                if len(data) == 1:
                    raise ValueError("legacy breakdown has no recognized fields")
                records.append(Record(data, source, line_no, "legacy"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise DiagnosticError(f"{source}:{line_no}: {exc}") from exc
    return records


@dataclass
class RequestGroup:
    label: str
    records: list[int] = field(default_factory=list)


@dataclass
class Report:
    records: list[Record]
    groups: list[RequestGroup]
    owners: dict[int, set[int]]
    warnings: list[str]


def associate(records: list[Record]) -> Report:
    """Associate explicit IDs; never guess across reused request IDs or clocks.

    Step IDs are many-to-many batch membership, not request identity. Task/trace
    ownership is propagated after reading *all* records so late completions stay
    attached to their original task, even after a request terminates.
    """
    unique = []
    seen: dict[str, Record] = {}
    warnings = []
    for record in records:
        key = json.dumps([record.kind, record.data], sort_keys=True)
        if key in seen:
            warnings.append(f"duplicate record {record.location}; first at "
                            f"{seen[key].location}; counted once")
        else:
            seen[key] = record
            unique.append(record)
    records = unique
    groups: list[RequestGroup] = []
    frontend: dict[str, int] = {}
    requests: dict[str, set[int]] = defaultdict(set)
    scoped_requests: dict[tuple[str, str], int] = {}
    for record in records:
        data = record.data
        fid, rid = data.get("frontend_id"), data.get("request_id")
        if fid:
            if fid not in frontend:
                frontend[fid] = len(groups)
                groups.append(RequestGroup(f"frontend={fid}"))
            if rid:
                requests[rid].add(frontend[fid])
    # Requests with no frontend binding still get an explicitly scoped group.
    for record in records:
        data = record.data
        rid = data.get("request_id")
        if rid and not requests[rid]:
            scope = str(data.get("run_id", "unknown"))
            key = (scope, rid)
            if key not in scoped_requests:
                scoped_requests[key] = len(groups)
                groups.append(RequestGroup(f"request={rid} run={scope}"))
    for (_, rid), group in scoped_requests.items():
        requests[rid].add(group)
    for rid, candidates in requests.items():
        if len(candidates) > 1:
            warnings.append(f"ambiguous/reused request_id={rid}: "
                            f"{len(candidates)} occurrences; no temporal guessing")
    owners: dict[int, set[int]] = {}
    links: dict[tuple[str, str], set[int]] = defaultdict(set)
    steps: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(records):
        data = record.data
        fid, rid = data.get("frontend_id"), data.get("request_id")
        candidates: set[int] = set()
        if fid:
            candidates = {frontend[fid]}
        elif rid:
            candidates = requests[rid].copy()
            if len(candidates) > 1:
                scoped = scoped_requests.get((str(data.get("run_id", "unknown")),
                                              rid))
                candidates = {scoped} if scoped is not None else set()
        owners[index] = candidates
        for name in STRONG_IDS:
            if data.get(name) and candidates:
                links[name, data[name]].update(candidates)
    # Fixed point supports trace -> task -> request chains in any file order.
    changed = True
    while changed:
        changed = False
        for index, record in enumerate(records):
            data = record.data
            linked = set()
            for name in STRONG_IDS:
                if data.get(name):
                    linked.update(links[name, data[name]])
            if linked and linked != owners[index]:
                if owners[index] and not owners[index].issubset(linked):
                    linked.update(owners[index])
                owners[index] = linked
                changed = True
            for name in STRONG_IDS:
                if data.get(name):
                    old = len(links[name, data[name]])
                    links[name, data[name]].update(owners[index])
                    changed |= len(links[name, data[name]]) != old
    for (name, value), candidates in links.items():
        if len(candidates) > 1:
            warnings.append(f"ambiguous {name}={value}: conflicting request owners")
    for index, record in enumerate(records):
        if record.data.get("step_id"):
            steps[record.data["step_id"]].update(owners[index])
    for index, record in enumerate(records):
        data = record.data
        if not owners[index] and data.get("step_id") and not any(
            data.get(name) for name in ("frontend_id", "request_id", "task_id",
                                       "trace_id")
        ):
            owners[index] = steps[data["step_id"]].copy()
        for group in owners[index]:
            groups[group].records.append(index)
        if not owners[index]:
            warnings.append(f"unassociated {record.event} at {record.location}: "
                            "missing or ambiguous request linkage")
    return Report(records, groups, owners, warnings)


def _text(value: Any) -> str:
    # Escaping user-controlled IDs prevents terminal control/newline injection.
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _ms(value: Any, divisor: float = 1_000_000) -> str:
    return f"{value / divisor:.3f} ms" if _number(value) else "unknown"


def _scope(data: dict[str, Any]) -> str:
    return " ".join(f"{key}={_text(data[key])}" for key in (
        "role", "host", "node", "rank", "pid", "run_id"
    ) if key in data) or "role/node/rank=unknown"


def _span_warnings(records: list[Record]) -> list[str]:
    starts: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    ends: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    warnings = []
    for record in records:
        data = record.data
        span = data.get("span_id")
        if span:
            key = (str(data.get("host", "unknown")),
                   str(data.get("run_id", "unknown")), span)
            (starts if record.event.endswith("_start") else ends)[key].append(record)
    for key in starts.keys() | ends.keys():
        opened, closed = starts[key], ends[key]
        if len(opened) != 1 or len(closed) != 1:
            warnings.append(f"incomplete/duplicate span_id={key[2]} "
                            f"starts={len(opened)} finishes={len(closed)}")
        for record in closed:
            data = record.data
            if not all(name in data for name in ("start_ns", "end_ns", "duration_ns")):
                warnings.append(f"missing measured interval for span_id={key[2]}")
            elif data["end_ns"] < data["start_ns"] or data["duration_ns"] < 0:
                warnings.append(f"invalid measured interval for span_id={key[2]}")
            parent = data.get("parent_span_id")
            if parent and (key[0], key[1], parent) not in ends:
                warnings.append(f"missing parent_span_id={parent} for span_id={key[2]}")
    return sorted(warnings)


def render_report(report: Report) -> str:
    lines = ["TTFT diagnostic report",
             f"Records: {len(report.records)}; request occurrences: {len(report.groups)}",
             "Times: Python ns converted to ms; native phases use explicit unit suffixes.",
             "Missing != zero. Shared steps, nested/parallel spans and ranks are NOT summed.",
             "Intervals are explanatory host measurements, not a critical path; clocks "
             "across nodes/processes are not aligned."]
    warnings = report.warnings + _span_warnings(report.records)
    for group_no, group in enumerate(report.groups):
        lines.append(f"\nRequest {group_no + 1}: {group.label}")
        group_records = [report.records[i] for i in group.records]
        events = {record.event for record in group_records}
        terminal = events & {"request_end", "request_core_end"}
        first = "request_first_token" in events or "legacy_breakdown" in events
        lines.append("  first token: " + ("observed" if first else "unknown/not observed")
                     + "; terminal: " + ("observed" if terminal else "missing/incomplete"))
        if not any(r.event == "request_match" for r in group_records):
            lines.append("  cache hit facts: unknown (no request_match)")
        if not any(r.event == "store_save_check" for r in group_records):
            lines.append("  save decision: unknown (no store_save_check)")
        scopes: dict[str, list[Record]] = defaultdict(list)
        for record in group_records:
            scopes[_scope(record.data)].append(record)
        for scope, records in scopes.items():
            lines.append(f"  Scope {scope}")
            for record in records:
                _render_record(record, lines, "    ")
            # Only detect overlap within one exact process/rank scope, not across clocks.
            intervals = [r for r in records if not r.event.endswith("_start")
                         and _number(r.data.get("start_ns"))
                         and _number(r.data.get("end_ns"))]
            intervals.sort(key=lambda r: r.data["start_ns"])
            latest = None
            for record in intervals:
                if latest and record.data["start_ns"] < latest.data["end_ns"]:
                    lines.append("    overlapping/nested measured intervals: "
                                 f"{latest.event} / {record.event}; not additive")
                if latest is None or record.data["end_ns"] > latest.data["end_ns"]:
                    latest = record
    unassociated = [r for i, r in enumerate(report.records) if not report.owners[i]]
    if unassociated:
        lines.append("\nUnassociated records (not assigned by temporal proximity):")
        for record in unassociated:
            lines.append(f"  Scope {_scope(record.data)}")
            _render_record(record, lines, "    ")
    if warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {_text(warning)}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _render_record(record: Record, lines: list[str], indent: str) -> None:
    data = record.data
    identifiers = " ".join(f"{name}={_text(data[name])}" for name in
                           IDENTIFIERS + ("span_id", "parent_span_id")
                           if data.get(name))
    lines.append(f"{indent}{record.event}" + (f" {identifiers}" if identifiers else ""))
    detail = indent + "  "
    if record.kind == "native":
        lines.append(f"{detail}scope: {data['native_scope']}")
        for name, value in data.items():
            divisor = (1_000_000 if name.endswith("_ns") else 1_000
                       if name.endswith("_us") else 1 if name.endswith("_ms")
                       else None)
            if divisor is not None:
                lines.append(f"{detail}{name}: {_ms(value, divisor)}")
        if "bytes" in data:
            lines.append(f"{detail}logical bytes: {data['bytes']:g} (not wire bytes)")
        lines.append(f"{detail}wire throughput: unknown (not measured)")
        if not data.get("trace_id"):
            lines.append(f"{detail}native correlation: missing trace_id")
    if record.event.endswith("_start"):
        lines.append(f"{detail}span opened; duration unknown until finish")
    elif "span_id" in data or "duration_ns" in data:
        lines.append(f"{detail}duration: {_ms(data.get('duration_ns'))}")
        if "start_ns" in data and "end_ns" in data:
            lines.append(f"{detail}monotonic interval: {data['start_ns']}.."
                         f"{data['end_ns']} ns (same-process only)")
        else:
            lines.append(f"{detail}measured interval: unknown/incomplete")
    facts = [f"{name}={_text(data[name])}" for name in FACT_FIELDS if name in data]
    # Include finalized diagnostic counters without hardcoding every step field.
    facts.extend(f"{name}={_text(value)}" for name, value in data.items()
                 if name not in FACT_FIELDS and (
                     name.endswith(("_tokens", "_count", "_keys", "_bytes"))
                     or name.startswith("num_")
                 ) and _number(value))
    if facts:
        lines.append(detail + "; ".join(facts))
    if record.kind == "legacy" or record.event == "request_first_token":
        values = {name + "_s": data[name + "_s"] for name in LEGACY_FIELDS
                  if name + "_s" in data}
        breakdown = data.get("breakdown")
        if isinstance(breakdown, dict):
            values.update({name + "_s": breakdown[name] for name in LEGACY_FIELDS
                           if _number(breakdown.get(name))})
        for name, value in values.items():
            label = "raw legacy residual" if name == "residual_s" else name
            lines.append(f"{detail}{label}: {_ms(value, 0.001)}")
        if "residual_s" not in values:
            lines.append(f"{detail}raw legacy residual: unknown (not recorded)")
        lines.append(f"{detail}residual is not recomputed by subtracting overlapping spans")


def preflight_report(store: Any = None) -> str:
    """Inspect this process only; optionally inspect an already-created Store.

    A standalone CLI cannot attest another running process's logger/client state.
    The supplied store is never constructed or initialized by this function.
    """
    lines = ["TTFT diagnostic preflight (this process only)",
             f"sys.executable: {sys.executable}",
             "VLLM_TTFT_DIAGNOSTICS="
             + repr(os.environ.get("VLLM_TTFT_DIAGNOSTICS", "<unset>"))]
    helper = sys.modules.get("vllm.v1.metrics.ttft_diagnostics")
    if helper is not None and callable(getattr(helper, "enabled", None)):
        lines.append(f"loaded helper enabled: {helper.enabled()}")
    else:
        lines.append("loaded helper enabled: untested (module not loaded; "
                     "environment alone does not prove startup configuration)")
    for name in ("vllm", "vllm_ascend", "mooncake"):
        module = sys.modules.get(name)
        if module is not None:
            lines.append(f"module {name}: loaded {getattr(module, '__file__', None)}")
        else:
            try:
                spec = importlib.machinery.PathFinder.find_spec(name)
                origin = spec.origin if spec is not None else "not found"
            except (ImportError, AttributeError, ValueError) as exc:
                origin = f"resolution failed: {type(exc).__name__}"
            lines.append(f"module {name}: not loaded; resolved path={origin}")
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith(("vllm.v1.metrics.ttft_diagnostics", "mooncake.",
                            "vllm_ascend.distributed.kv_transfer")) and module:
            lines.append(f"loaded module {name}: {getattr(module, '__file__', None)}")
    for name in ("", "vllm", "vllm.v1.metrics.ttft_diagnostics",
                 "vllm_ascend", "mooncake"):
        logger = logging.getLogger(name)
        lines.append(f"logger {name or '<root>'}: effective="
                     f"{logging.getLevelName(logger.getEffectiveLevel())} "
                     f"disabled={logger.disabled} propagate={logger.propagate}")
        current = logger
        while current is not None:
            for handler in current.handlers:
                lines.append(f"  handler {type(handler).__module__}."
                             f"{type(handler).__name__}: "
                             f"level={logging.getLevelName(handler.level)} "
                             f"filters={len(handler.filters)} "
                             f"via={current.name or '<root>'}")
            if not current.propagate:
                break
            current = current.parent
        if not logger.hasHandlers():
            lines.append("  handlers: none; lastResort may handle WARNING+")
    lines.append(f"logging global disable threshold: {logging.root.manager.disable}")
    if store is None:
        lines.extend([
            "native traced method: untested (no initialized Store supplied)",
            "native RealClient trace capability: uninitialized/untested",
        ])
    else:
        method = getattr(store, "batch_put_from_multi_buffers_with_trace", None)
        capability = getattr(store, "supports_batch_put_trace", None)
        lines.append(f"native Store type: {type(store).__module__}.{type(store).__name__}")
        lines.append(f"native traced method present: {callable(method)}")
        if callable(capability):
            try:
                lines.append(f"native RealClient trace capability: {capability()!r}")
            except Exception as exc:
                lines.append(f"native RealClient trace capability: untested "
                             f"({type(exc).__name__})")
        else:
            lines.append("native RealClient trace capability: unavailable/untested")
    lines.append("No model/NPU/Store initialized. Native logging and another process's "
                 "configuration are untested; verify startup diagnostics there.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, stdin: TextIO | None = None,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report_parser = commands.add_parser("report", help="summarize diagnostic log files")
    report_parser.add_argument("paths", nargs="*", default=[],
                               help="UTF-8 log paths; '-' or no paths reads stdin")
    commands.add_parser("preflight", help="inspect configuration without initialization")
    args = parser.parse_args(argv)
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    stdin = stdin if stdin is not None else sys.stdin
    if args.command == "preflight":
        stdout.write(preflight_report())
        return 0
    try:
        records = []
        used_stdin = False
        for path in args.paths or ["-"]:
            if path == "-":
                if used_stdin:
                    raise DiagnosticError("stdin '-' may only be specified once")
                records.extend(parse_lines(stdin))
                used_stdin = True
            else:
                with Path(path).open(encoding="utf-8-sig") as stream:
                    records.extend(parse_lines(stream, path))
        if not records:
            raise DiagnosticError("no recognized TTFT diagnostic records found")
        stdout.write(render_report(associate(records)))
        return 0
    except (OSError, UnicodeError, DiagnosticError) as exc:
        stderr.write(f"ttft_diagnostics: error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
