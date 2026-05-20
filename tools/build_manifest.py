#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a bootstrap Phase D engine manifest.

This is the first implementation step for the v3.6.1 manifest toolchain.  It
creates a deterministic manifest shell that can be enriched by later engine
reflection code.

Usage:
  python tools/build_manifest.py --engine vllm --version 0.11.0 --output auto
  python tools/build_manifest.py --engine vllm --version 0.11.0 --from-json seed.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

from _config_tooling import SUPPORTED_ENGINES, load_structured_file, write_structured_file

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _empty_manifest(engine: str, version: str, generated_from: str) -> dict[str, Any]:
    return {
        "engine": engine,
        "version": str(version),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "generated_from": generated_from,
        "cli_params": {},
        "env_vars": {},
    }


def build_manifest(engine: str, version: str, *, generated_from: str = "manual-bootstrap", seed: dict[str, Any] | None = None) -> dict[str, Any]:
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"unsupported engine: {engine}")
    manifest = _empty_manifest(engine, version, generated_from)
    if seed:
        for key in ("base_engine", "base_version", "plugin_version", "cann_version_range", "torch_npu_version_range"):
            if key in seed:
                manifest[key] = seed[key]
        for section in ("cli_params", "env_vars", "additional_config", "generated_file"):
            value = seed.get(section)
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"seed.{section} must be an object")
                manifest[section] = value
    return manifest


def _resolve_output(config_root: Path, engine: str, version: str, output: str) -> Path:
    if output == "auto":
        return config_root / "manifests" / engine / f"{version}.yaml"
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bootstrap engine manifest.")
    parser.add_argument("--engine", required=True, choices=sorted(SUPPORTED_ENGINES))
    parser.add_argument("--version", required=True)
    parser.add_argument("--config-root", default=str(_REPO_ROOT / "wings_control" / "config"))
    parser.add_argument("--output", default="auto", help="Output path, or 'auto' for config/manifests/<engine>/<version>.yaml")
    parser.add_argument("--generated-from", default="manual-bootstrap")
    parser.add_argument("--from-json", default="", help="Optional JSON/YAML seed with cli_params/env_vars sections")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    seed = load_structured_file(args.from_json) if args.from_json else None
    manifest = build_manifest(args.engine, args.version, generated_from=args.generated_from, seed=seed)
    out = _resolve_output(config_root, args.engine, args.version, args.output)
    write_structured_file(out, manifest)
    print(out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
