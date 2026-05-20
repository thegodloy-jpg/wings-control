#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lint Phase D deviations files.

Usage:
  python tools/lint_deviations.py [--config-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _config_tooling import (
    SUPPORTED_TARGETS,
    iter_config_files,
    load_structured_file,
    print_results,
    require_fields,
    validate_hardware_list,
    validate_source_type,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def validate_deviation_document(path: str | Path, data: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    deviations = data.get("deviations", [])
    if deviations is None:
        deviations = []
    if not isinstance(deviations, list):
        return [("error", "top-level deviations must be a list")]

    for index, item in enumerate(deviations):
        label = f"deviations[{index}]"
        if not isinstance(item, dict):
            diagnostics.append(("error", f"{label} must be an object"))
            continue
        diagnostics.extend(require_fields(
            item,
            ["field", "target", "wings_value", "applies_to_versions", "rationale", "source_type", "decision_date"],
            label,
        ))
        target = item.get("target")
        if target and target not in SUPPORTED_TARGETS:
            diagnostics.append(("error", f"{label}.target invalid: {target!r}"))
        diagnostics.extend(validate_source_type(item, label))
        diagnostics.extend(validate_hardware_list(item, "applies_to_hardware", label))
        if "applies_to_hardware" not in item:
            diagnostics.append(("warning", f"{label} has no applies_to_hardware; deviation applies globally"))
        if item.get("upstream_default") == item.get("wings_value"):
            diagnostics.append(("error", f"{label} wings_value equals upstream_default"))
    return diagnostics


def lint_all(config_root: Path, output_json: bool) -> int:
    files = iter_config_files(config_root / "deviations")
    results: list[dict[str, Any]] = []
    for path in files:
        try:
            data = load_structured_file(path)
            diagnostics = validate_deviation_document(path, data)
        except Exception as exc:
            diagnostics = [("error", f"failed to load: {exc}")]
        results.append({
            "file": str(path),
            "diagnostics": [{"level": level, "message": message} for level, message in diagnostics],
        })
    return print_results(results, output_json=output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint wings-control deviations files.")
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
