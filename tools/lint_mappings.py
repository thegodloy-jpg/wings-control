#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lint Phase D canonical-to-engine mappings.

Usage:
  python tools/lint_mappings.py [--config-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _config_tooling import (
    SUPPORTED_ENGINES,
    SUPPORTED_TARGETS,
    iter_config_files,
    load_structured_file,
    print_results,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _manifest_fields(config_root: Path, engine: str) -> set[str]:
    fields: set[str] = set()
    manifest_dir = config_root / "manifests" / engine
    for path in iter_config_files(manifest_dir):
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        for section in ("cli_params", "env_vars", "additional_config", "generated_file"):
            values = data.get(section) or {}
            if isinstance(values, dict):
                fields.update(str(key) for key in values.keys())
    return fields


def validate_mapping_document(path: str | Path, data: dict[str, Any], *, config_root: str | Path | None = None) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    canonical = data.get("canonical")
    if not isinstance(canonical, dict):
        return [("error", "top-level canonical must be an object")]

    root = Path(config_root) if config_root else None
    manifest_cache: dict[str, set[str]] = {}
    for canonical_key, engine_map in canonical.items():
        label = f"canonical.{canonical_key}"
        if not isinstance(engine_map, dict):
            diagnostics.append(("error", f"{label} must be an object"))
            continue
        if not engine_map:
            diagnostics.append(("warning", f"{label} has no engine mappings"))
        for engine, mapping in engine_map.items():
            item_label = f"{label}.{engine}"
            if engine not in SUPPORTED_ENGINES:
                diagnostics.append(("error", f"{item_label} invalid engine"))
                continue
            if not isinstance(mapping, dict):
                diagnostics.append(("error", f"{item_label} must be an object"))
                continue
            for field in ("field", "target", "applies_to_versions"):
                if not mapping.get(field):
                    diagnostics.append(("error", f"{item_label} missing {field}"))
            target = mapping.get("target")
            if target and target not in SUPPORTED_TARGETS:
                diagnostics.append(("error", f"{item_label}.target invalid: {target!r}"))
            if root is not None:
                manifest_fields = manifest_cache.setdefault(engine, _manifest_fields(root, engine))
                mapped_field = str(mapping.get("field") or "")
                if manifest_fields and mapped_field and mapped_field not in manifest_fields:
                    diagnostics.append(("warning", f"{item_label}.field {mapped_field!r} not found in manifests/{engine}"))
    return diagnostics


def lint_all(config_root: Path, output_json: bool) -> int:
    files = iter_config_files(config_root / "mappings")
    results: list[dict[str, Any]] = []
    for path in files:
        try:
            data = load_structured_file(path)
            diagnostics = validate_mapping_document(path, data, config_root=config_root)
        except Exception as exc:
            diagnostics = [("error", f"failed to load: {exc}")]
        results.append({
            "file": str(path),
            "diagnostics": [{"level": level, "message": message} for level, message in diagnostics],
        })
    return print_results(results, output_json=output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint wings-control canonical mapping files.")
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
