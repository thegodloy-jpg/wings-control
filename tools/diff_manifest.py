#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diff two Phase D engine manifests.

Compares manifest sections used by lint/baseline checks and reports added,
removed, and changed defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _config_tooling import load_structured_file

COMPARE_SECTIONS = ("cli_params", "env_vars", "additional_config", "generated_file")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name) or {}
    return value if isinstance(value, dict) else {}


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "old": {"engine": old.get("engine"), "version": old.get("version")},
        "new": {"engine": new.get("engine"), "version": new.get("version")},
        "sections": {},
        "summary": {"added": 0, "removed": 0, "changed": 0},
    }
    for section_name in COMPARE_SECTIONS:
        left = _section(old, section_name)
        right = _section(new, section_name)
        added = sorted(set(right) - set(left))
        removed = sorted(set(left) - set(right))
        changed: dict[str, Any] = {}
        for key in sorted(set(left) & set(right)):
            old_default = left.get(key, {}).get("upstream_default") if isinstance(left.get(key), dict) else None
            new_default = right.get(key, {}).get("upstream_default") if isinstance(right.get(key), dict) else None
            old_type = left.get(key, {}).get("type") if isinstance(left.get(key), dict) else None
            new_type = right.get(key, {}).get("type") if isinstance(right.get(key), dict) else None
            if old_default != new_default or old_type != new_type:
                changed[key] = {
                    "old": {"type": old_type, "upstream_default": old_default},
                    "new": {"type": new_type, "upstream_default": new_default},
                }
        result["sections"][section_name] = {
            "added": added,
            "removed": removed,
            "changed": changed,
        }
        result["summary"]["added"] += len(added)
        result["summary"]["removed"] += len(removed)
        result["summary"]["changed"] += len(changed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two engine manifest files.")
    parser.add_argument("old_manifest")
    parser.add_argument("new_manifest")
    parser.add_argument("--fail-on-change", action="store_true", help="Exit 1 when any diff is detected")
    args = parser.parse_args()

    old = load_structured_file(args.old_manifest)
    new = load_structured_file(args.new_manifest)
    diff = diff_manifests(old, new)
    print(json.dumps(diff, ensure_ascii=False, indent=2))
    changed = sum(diff["summary"].values())
    if args.fail_on_change and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
