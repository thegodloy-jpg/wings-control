#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lint Phase D env_policies files.

Usage:
  python tools/lint_env_policies.py [--config-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from _config_tooling import (
    SUPPORTED_ENV_MODES,
    iter_config_files,
    load_structured_file,
    print_results,
    require_fields,
    validate_hardware_list,
    validate_source_type,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_env_policy_document(path: str | Path, data: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    policies = data.get("env_policies", [])
    if policies is None:
        policies = []
    if not isinstance(policies, list):
        return [("error", "top-level env_policies must be a list")]

    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(policies):
        label = f"env_policies[{index}]"
        if not isinstance(item, dict):
            diagnostics.append(("error", f"{label} must be an object"))
            continue
        diagnostics.extend(require_fields(item, ["name", "mode", "applies_to_versions"], label))
        name = str(item.get("name") or "")
        if name and not _ENV_NAME_RE.match(name):
            diagnostics.append(("error", f"{label}.name is not a valid env var: {name!r}"))
        applies_when = item.get("applies_when") or {}
        if applies_when and not isinstance(applies_when, dict):
            diagnostics.append(("error", f"{label}.applies_when must be an object"))
            applies_when = {}
        duplicate_key = (name, json.dumps(applies_when, ensure_ascii=False, sort_keys=True))
        if duplicate_key in seen:
            suffix = f" with applies_when={applies_when}" if applies_when else ""
            diagnostics.append(("error", f"duplicate env policy for {name}{suffix}"))
        seen.add(duplicate_key)
        mode = item.get("mode")
        if mode not in SUPPORTED_ENV_MODES:
            diagnostics.append(("error", f"{label}.mode invalid: {mode!r}"))
            continue
        diagnostics.extend(validate_hardware_list(item, "applies_to_hardware", label))
        if "value_template" in item and not isinstance(item.get("value_template"), str):
            diagnostics.append(("error", f"{label}.value_template must be a string"))
        if mode == "idempotent" and not ("default_value" in item or item.get("derive_from") or item.get("value_template")):
            diagnostics.append(("error", f"{label} mode=idempotent requires default_value, value_template, or derive_from"))
        if mode == "force_override" and not (item.get("rationale") or item.get("derive_from")):
            diagnostics.append(("error", f"{label} mode=force_override requires rationale or derive_from"))
        if mode == "force_override":
            diagnostics.extend(validate_source_type(item, label))
            if not item.get("decision_date"):
                diagnostics.append(("error", f"{label} mode=force_override missing decision_date"))
        elif item.get("source_type"):
            diagnostics.extend(validate_source_type(item, label))
        if mode == "inherit" and "default_value" in item:
            diagnostics.append(("warning", f"{label} mode=inherit ignores default_value"))
        if mode == "inherit" and "value_template" in item:
            diagnostics.append(("warning", f"{label} mode=inherit ignores value_template"))
    return diagnostics


def lint_all(config_root: Path, output_json: bool) -> int:
    files = iter_config_files(config_root / "env_policies")
    results: list[dict[str, Any]] = []
    for path in files:
        try:
            data = load_structured_file(path)
            diagnostics = validate_env_policy_document(path, data)
        except Exception as exc:
            diagnostics = [("error", f"failed to load: {exc}")]
        results.append({
            "file": str(path),
            "diagnostics": [{"level": level, "message": message} for level, message in diagnostics],
        })
    return print_results(results, output_json=output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint wings-control env_policies files.")
    parser.add_argument("--config-root", default=str(_REPO_ROOT / "wings_control" / "config"))
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    if not config_root.is_dir():
        print(f"ERROR: config root does not exist: {config_root}", file=sys.stderr)
        sys.exit(1)
    sys.exit(lint_all(config_root, args.output_json))


if __name__ == "__main__":
    main()
