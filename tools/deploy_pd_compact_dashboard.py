#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage an isolated Grafana snapshot config; never start or stop a service."""

import argparse
import json
from pathlib import Path


def stage(output: Path, api_url: str, plugins: Path | None = None) -> Path:
    output = output.resolve()
    provisioning = output / "provisioning"
    for path in (
        output / "data",
        output / "logs",
        output / "dashboards",
        provisioning / "datasources",
        provisioning / "dashboards",
    ):
        path.mkdir(parents=True, exist_ok=True)
    dashboard = Path(__file__).with_name("grafana_pd_compact_dashboard.json")
    (output / "dashboards" / dashboard.name).write_text(
        dashboard.read_text(encoding="utf-8"), encoding="utf-8"
    )
    datasource = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "PD Compact Rows",
                "uid": "pd-compact-rows",
                "type": "marcusolsson-json-datasource",
                "access": "proxy",
                "url": api_url,
                "isDefault": True,
                "editable": False,
            }
        ],
    }
    providers = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "PD Compact schema 2",
                "orgId": 1,
                "folder": "PD Compact",
                "type": "file",
                "disableDeletion": False,
                "editable": True,
                "options": {"path": (output / "dashboards").as_posix()},
            }
        ],
    }
    # JSON is valid YAML and avoids an external YAML dependency.
    (provisioning / "datasources" / "compact.yaml").write_text(
        json.dumps(datasource, indent=2) + "\n", encoding="utf-8"
    )
    (provisioning / "dashboards" / "compact.yaml").write_text(
        json.dumps(providers, indent=2) + "\n", encoding="utf-8"
    )
    config = (
        "[server]\nhttp_addr = 127.0.0.1\nhttp_port = 3033\n"
        "[paths]\n"
        f"data = {(output / 'data').as_posix()}\n"
        f"logs = {(output / 'logs').as_posix()}\n"
        f"provisioning = {provisioning.as_posix()}\n"
    )
    if plugins is not None:
        config += f"plugins = {plugins.resolve().as_posix()}\n"
    config += "[analytics]\nreporting_enabled = false\ncheck_for_updates = false\n"
    path = output / "grafana.ini"
    path.write_text(config, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8766")
    parser.add_argument("--plugins", type=Path)
    args = parser.parse_args()
    path = stage(args.out, args.api_url, args.plugins)
    print(f"Isolated Grafana port 3033 config: {path}")
    print("Start the existing Grafana binary with --config pointing at this file.")


if __name__ == "__main__":
    main()
