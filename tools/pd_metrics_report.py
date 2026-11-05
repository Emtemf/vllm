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
from collections import defaultdict
from pathlib import Path
from typing import Any

MARKER = "[PD_DIAG] "


def clock(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(k) for k in ("host", "pid", "run_id"))


def read_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                try:
                    record = json.loads(line.split(MARKER, 1)[1])
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
    return {
        "schema_version": 1,
        "complete": False,
        "completeness_note": (
            "Observed evidence only; log presence does not prove complete capture. "
            "Shared batch ranges are not exclusive request latency; parallel TP "
            "ranges and cross-process clocks are never added. Bytes retain their "
            "producer semantics and are not inferred wire throughput."
        ),
        "parse_errors": errors,
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


def render_html(report: dict[str, Any]) -> str:
    def escape(value: Any) -> str:
        return html.escape(str(value))

    def evidence(records: Any) -> str:
        text = json.dumps(records, ensure_ascii=False, indent=2)
        return "<pre>" + escape(text) + "</pre>"

    def table(headers: tuple[str, ...], rows: list[list[Any]]) -> str:
        return (
            "<table><thead><tr>"
            + "".join("<th>" + escape(h) + "</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(
                    "<td>" + escape("unknown" if c is None else c) + "</td>"
                    for c in row
                )
                + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    out = [
        '<!doctype html><html><meta charset="utf-8"><title>P/D metrics</title>',
        (
            "<style>body{font-family:system-ui;margin:24px}"
            "table{border-collapse:collapse;display:block;overflow:auto}"
            "th,td{border:1px solid #888;padding:8px;vertical-align:top}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
            "details{margin:8px}</style>"
        ),
        "<h1>P/D metrics</h1><p>" + escape(report["completeness_note"]) + "</p>",
    ]
    if report["parse_errors"]:
        out.append("<h2>Invalid records</h2>" + evidence(report["parse_errors"]))
    fields = (
        "request_id",
        "first_token_observed",
        "time_to_first_token",
        "e2e_latency",
        "queued_time",
        "prefill_time",
        "decode_time",
        "mean_time_per_output_token",
        "num_prompt_tokens",
        "num_generation_tokens",
        "num_preemptions",
    )
    summaries = [
        record
        for request in report["requests"]
        for record in request["observations"]
        if record["event"] == "request_summary"
    ]
    out.append("<h2>Native request statistics (seconds)</h2>")
    out.append(
        "<p>Native timing formulas are unchanged. A first_token_observed value "
        "of false means the request produced no token; its native timestamp "
        "must not be treated as an observed first-token latency.</p>"
    )
    out.append(table(fields, [[r.get(f) for f in fields] for r in summaries]))
    out.append("<details><summary>Explicit identity links</summary>")
    out.append(evidence(report["identity_links"]) + "</details>")
    for request in report["requests"]:
        out.append(
            "<details><summary>Request "
            + escape(request["identity"])
            + "</summary>"
            + evidence(request["observations"])
            + "</details>"
        )
    out.append("<h2>Executions: shared scheduler evidence and separate TP rows</h2>")
    for execution in report["executions"]:
        out.append("<h3>" + escape(execution["execution_id"]) + "</h3>")
        out.append("<p>Members: " + escape(execution["request_ids"]) + "</p>")
        out.append("<details><summary>Shared batch evidence</summary>")
        out.append(evidence(execution["shared_observations"]) + "</details>")
        rows = []
        for rank in execution["ranks"]:
            for event, value in rank["event_union_ns"].items():
                rows.append(
                    [
                        rank["domain"],
                        event,
                        None if value is None else round(value / 1e6, 6),
                    ]
                )
            out.append(
                "<details><summary>TP evidence "
                + escape(rank["domain"])
                + "</summary>"
                + evidence(rank["observations"])
                + "</details>"
            )
        out.append(table(("Host / PID / run / TP", "Event", "Host ms"), rows))
    out.append("<h2>Transfer / storage tasks</h2>")
    for task in report["tasks"]:
        out.append("<h3>" + escape(task["task_id"]) + "</h3>")
        rows = []
        for record in task["observations"]:
            for name, stage in record.get("stages", {}).items():
                ns = stage.get("duration_ns")
                rows.append(
                    [
                        name,
                        stage.get("calls"),
                        None if ns is None else ns / 1e6,
                        stage.get("exceptions"),
                    ]
                )
        if rows:
            out.append(
                table(
                    (
                        "Stage (serial calls, not envelope)",
                        "Calls",
                        "Host ms",
                        "Errors",
                    ),
                    rows,
                )
            )
        out.append("<details><summary>Counters / result / identity evidence</summary>")
        out.append(evidence(task["observations"]) + "</details>")
    out.append(
        "<details><summary>Unassociated evidence</summary>"
        + evidence(report["unassociated"])
        + "</details></html>"
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
