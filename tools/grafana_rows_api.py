#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve a read-only compact schema-2 snapshot (isolated default port 8766)."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from pd_grafana_json import ARRAYS, selected_request


def validate_payload(payload: dict) -> None:
    if payload.get("version") != 2:
        raise ValueError("schema-2 rows required")
    for name in ARRAYS:
        if not isinstance(payload.get(name), list):
            raise ValueError("missing canonical array: " + name)


def rows_for(payload: dict, path: str) -> list[dict]:
    parsed = urlparse(path)
    endpoint = parsed.path.strip("/")
    if endpoint not in (*ARRAYS, "node_options", "sides", "links"):
        raise KeyError(endpoint)
    query = parse_qs(parsed.query)
    request_key = query.get("request_key", [None])[0]
    side = query.get("side", [None])[0]
    if side not in (None, "", "$__all", "P", "D", "unknown"):
        raise ValueError("side must be P, D, or unknown")
    name = (
        "nodes"
        if endpoint in ("node_options", "sides")
        else "transport_links"
        if endpoint == "links"
        else endpoint
    )
    rows = payload.get(name, [])
    if request_key not in (None, "", "$__all"):
        if endpoint == "links":
            rows = [
                row
                for row in rows
                if request_key
                in (row.get("source_request_key"), row.get("target_request_key"))
            ]
        else:
            rows = [
                row
                for row in rows
                if row.get("request_key") == request_key
                or (endpoint == "quality" and row.get("request_key") is None)
            ]
    if endpoint == "sides":
        return [
            {"side": value or "unknown"}
            for value in dict.fromkeys(row.get("side") for row in rows)
        ]
    if side not in (None, "", "$__all"):
        selected_side = None if side == "unknown" else side
        if endpoint == "requests":
            rows = [
                selected_request(row, selected_side)
                for row in rows
                if selected_side in row.get("sides", [])
            ]
        else:
            rows = [
                row
                for row in rows
                if row.get("side") == selected_side
                or (endpoint == "quality" and row.get("side") is None)
            ]
    if endpoint == "requests":
        rows = [
            {
                **row,
                "display": str(
                    row.get("request_id")
                    or "Observation "
                    + str(row.get("observation_id") or row["request_key"])
                ),
            }
            for row in rows
        ]
    if endpoint == "node_options":
        return [{"node_id": "__root", "path": "L1 groups"}] + [
            {field: row.get(field) for field in ("node_id", "path", "parent_id")}
            for row in rows
            if row.get("child_count", 0) > 0
        ]
    if endpoint == "nodes":
        parent_id = query.get("parent_id", [None])[0]
        if parent_id == "__root":
            rows = [row for row in rows if row.get("parent_id") is None]
        elif parent_id not in (None, "", "$__all"):
            rows = [row for row in rows if row.get("parent_id") == parent_id]
    return rows


class Handler(BaseHTTPRequestHandler):
    payload: ClassVar[dict] = {}

    def do_GET(self):
        try:
            rows = rows_for(self.payload, self.path)
            body, status = json.dumps(rows, ensure_ascii=False).encode(), 200
        except (KeyError, ValueError) as exc:
            body, status = json.dumps({"error": str(exc)}).encode(), 400
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    try:
        Handler.payload = json.loads(args.rows.read_text(encoding="utf-8"))
        validate_payload(Handler.payload)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Serving compact schema-2 rows on http://{args.host}:{args.port}", flush=True
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
