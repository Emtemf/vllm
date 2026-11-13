#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export canonical schema-2 P/D evidence as standard-library Grafana rows."""

import argparse
import json
import math
from pathlib import Path

ARRAYS = ("requests", "nodes", "observations", "tasks", "pd_transfers", "quality")


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def selected_request(row: dict, side: str | None) -> dict:
    """Select one official source, never sum/choose among repeated lifecycles."""
    result = dict(row)
    candidates = [
        item for item in row.get("official_summaries", []) if item.get("side") == side
    ]
    summary = candidates[0].get("official", {}) if len(candidates) == 1 else {}
    result.update(
        side=side,
        official=summary,
        official_state="measured"
        if len(candidates) == 1
        else "ambiguous"
        if candidates
        else "missing",
    )
    for target, source in (
        ("input_tokens", "num_prompt_tokens"),
        ("output_tokens", "num_generation_tokens"),
        ("cached_tokens", "num_cached_tokens"),
    ):
        result[target] = summary.get(source) if _number(summary.get(source)) else None
    # Native statistics use seconds. Do not infer TTFT from lifetime traces,
    # transfer envelopes, detokenization, first/last timestamps or stage sums.
    for target, source in (
        ("ttft_ms", "time_to_first_token"),
        ("e2e_ms", "e2e_latency"),
        ("decode_ms", "decode_time"),
    ):
        value = summary.get(source)
        observed = (
            source != "time_to_first_token"
            or summary.get("first_token_observed") is True
        )
        result[target] = (
            value * 1000
            if summary.get("unit") == "seconds" and _number(value) and observed
            else None
        )
    result["first_token_observed"] = summary.get("first_token_observed")
    result["official_source_observations"] = (
        candidates[0].get("source_observations", []) if len(candidates) == 1 else []
    )
    return result


def flatten(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("version") != 2:
        raise ValueError("schema-2 report required; no schema-1 reconstruction")
    dashboard = report.get("dashboard")
    if not isinstance(dashboard, dict) or dashboard.get("version") != 2:
        raise ValueError("canonical schema-2 dashboard required")
    for name in ARRAYS:
        if not isinstance(dashboard.get(name), list):
            raise ValueError("missing canonical array: " + name)
    # Nodes are exported verbatim; HTML and Grafana cannot disagree on a tree.
    return {
        "version": 2,
        "snapshot": True,
        "warning": report.get("warning"),
        **{name: dashboard[name] for name in ARRAYS},
        "transport_links": dashboard.get("transport_links", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--logs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    args = parser.parse_args()
    if (args.report is None) == (args.logs is None):
        parser.error("provide either a report JSON or --logs LOG [LOG ...]")
    try:
        if args.logs is not None:
            from pd_metrics_report import read_records, summarize

            records, errors = read_records(args.logs)
            if not records and not errors:
                errors.append("No schema-2 diagnostic records found")
            report = summarize(records, errors)
        else:
            report = json.loads(args.report.read_text(encoding="utf-8"))
        payload = flatten(report)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(
        f"{len(payload['requests'])} requests; "
        f"{len(payload['nodes'])} canonical nodes -> {args.out}"
    )


if __name__ == "__main__":
    main()
