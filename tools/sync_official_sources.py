#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sync wings-control configs against upstream official documentation.

Read-only by design: this tool fetches the URLs declared in
config/official_sources.yaml, diffs them against local manifests/inventory, and
writes JSON reports under build/official_start_command_refs/<engine>/<source_id>.json
for human review. It never modifies project yaml.

Typical usage::

    # First run (writes cache + reports):
    python tools/sync_official_sources.py --engines vllm vllm_ascend

    # Offline re-run from cache:
    python tools/sync_official_sources.py --engines vllm vllm_ascend --use-cache

    # Air-gapped: point at a directory of saved HTML fixtures:
    python tools/sync_official_sources.py --fixture-dir tests/fixtures/official_sources_html

The tool exits non-zero only when fetching fails for every engine. Drift itself
is informational — promotion to project yaml is a separate human-reviewed step
(see official_sources.yaml: sync_contract.promotion_requires).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _sync_official import SyncConfig, run_sync

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "build" / "official_start_command_refs"
_DEFAULT_CACHE = _REPO_ROOT / "build" / "official_start_command_refs" / "_raw"
_DEFAULT_CONFIG = _REPO_ROOT / "wings_control" / "config"
_DEFAULT_ENGINES = ["vllm", "vllm_ascend"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--engines",
        nargs="+",
        default=_DEFAULT_ENGINES,
        help=f"Engines to sync (default: {' '.join(_DEFAULT_ENGINES)}).",
    )
    parser.add_argument("--config-root", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    parser.add_argument(
        "--fixture-dir",
        default=None,
        help="Optional directory of saved HTML fixtures (offline mode).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip HTTP if cached HTML exists under --cache-dir.",
    )
    parser.add_argument("--json", action="store_true", help="Emit aggregate JSON instead of text summary.")
    return parser.parse_args(argv)


def _format_text(aggregate: dict) -> str:
    lines = ["Official source sync report"]
    summary = aggregate["summary"]
    lines.append(
        f"summary: reports={summary['reports']}, "
        f"fetch_errors={summary['fetch_errors']}, drift_total={summary['drift_total']}"
    )
    for engine, sources in aggregate["engines"].items():
        lines.append(f"\n[{engine}]")
        for entry in sources:
            err = f" ERROR: {entry['fetch_error']}" if entry["fetch_error"] else ""
            lines.append(
                f"  {entry['source_id']}: drift={entry['drift_count']} report={entry['report']}{err}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = SyncConfig(
        repo_root=_REPO_ROOT,
        config_root=Path(args.config_root),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        fixture_dir=Path(args.fixture_dir) if args.fixture_dir else None,
        use_cache=args.use_cache,
        engines=list(args.engines),
    )
    aggregate = run_sync(cfg)
    if args.json:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    else:
        print(_format_text(aggregate))
    if aggregate["summary"]["reports"] == 0:
        return 1
    if aggregate["summary"]["fetch_errors"] and aggregate["summary"]["fetch_errors"] == aggregate["summary"]["reports"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
