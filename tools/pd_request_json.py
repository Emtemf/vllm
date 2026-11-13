#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export schema-2 requests with their canonical nodes and original evidence."""

import argparse
import json
from pathlib import Path

from pd_grafana_json import selected_request
from pd_metrics_report import build_dashboard, read_records


def convert(records: list[dict]) -> list[dict]:
    dashboard = build_dashboard(records)
    output = []
    for request in dashboard["requests"]:
        key = request["request_key"]
        output.append(
            {
                "version": 2,
                **request,
                "official_by_side": [
                    selected_request(request, side) for side in request["sides"]
                ],
                **{
                    name: [
                        row for row in dashboard[name] if row.get("request_key") == key
                    ]
                    for name in (
                        "nodes",
                        "observations",
                        "tasks",
                        "pd_transfers",
                        "quality",
                    )
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        records, errors = read_records(args.logs)
        if errors or not records:
            raise ValueError(
                "Invalid diagnostic records:\n"
                + "\n".join(errors or ["No schema-2 diagnostic records found"])
            )
        output = convert(records)
        args.out.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(f"{len(output)} request/observation records -> {args.out}")


if __name__ == "__main__":
    main()
