#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recipe lint tool for wings-control (Phase A).

Scans all recipe YAML files under config/recipes/ and reports validation
diagnostics using the rules defined in recipe_loader.validate_recipe_document().

Exit codes:
  0 – no errors found (warnings may still be present)
  1 – at least one error found

Usage:
  python tools/lint_recipes.py [--config-root PATH] [--include-experimental]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root or from tools/ directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "wings_control"))

from core.hardware_detect import KNOWN_CHIP_IDS  # noqa: E402
from core.recipe_loader import load_structured_file, validate_recipe_document  # noqa: E402


def _iter_recipe_files(config_root: Path, include_experimental: bool) -> list[Path]:
    files: list[Path] = []
    for kind in ("architectures", "models"):
        base = config_root / "recipes" / kind
        if base.is_dir():
            files.extend(sorted(p for p in base.glob("*.yaml") if p.is_file()))
            files.extend(sorted(p for p in base.glob("*.json") if p.is_file()))
        if include_experimental:
            exp = base / "_experimental"
            if exp.is_dir():
                files.extend(sorted(p for p in exp.glob("*.yaml") if p.is_file()))
                files.extend(sorted(p for p in exp.glob("*.json") if p.is_file()))
    return files


def lint_all(config_root: Path, include_experimental: bool, output_json: bool) -> int:
    files = _iter_recipe_files(config_root, include_experimental)
    if not files:
        msg = f"[lint_recipes] no recipe files found under {config_root / 'recipes'}"
        if output_json:
            print(json.dumps({"files_checked": 0, "errors": 0, "warnings": 0, "results": [], "message": msg}))
        else:
            print(msg, file=sys.stderr)
        return 0

    total_errors = 0
    total_warnings = 0
    results = []

    for path in files:
        try:
            data = load_structured_file(path)
        except Exception as exc:
            results.append({
                "file": str(path),
                "diagnostics": [{"level": "error", "message": f"failed to load: {exc}"}],
            })
            total_errors += 1
            continue

        diagnostics = validate_recipe_document(path, data, known_chips=set(KNOWN_CHIP_IDS))
        file_errors = sum(1 for lvl, _ in diagnostics if lvl == "error")
        file_warnings = sum(1 for lvl, _ in diagnostics if lvl == "warning")
        total_errors += file_errors
        total_warnings += file_warnings
        results.append({
            "file": str(path),
            "diagnostics": [{"level": lvl, "message": msg} for lvl, msg in diagnostics],
        })

    if output_json:
        print(json.dumps({
            "files_checked": len(files),
            "errors": total_errors,
            "warnings": total_warnings,
            "results": results,
        }, indent=2))
    else:
        for r in results:
            rel = Path(r["file"])
            try:
                rel = Path(r["file"]).relative_to(config_root.parent)
            except ValueError:
                pass
            for d in r["diagnostics"]:
                prefix = "ERROR" if d["level"] == "error" else "WARN "
                print(f"{prefix}  {rel}: {d['message']}")
        print(f"\n{len(files)} file(s) checked — {total_errors} error(s), {total_warnings} warning(s)")

    return 1 if total_errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint wings-control recipe YAML files.")
    parser.add_argument(
        "--config-root",
        default=str(_REPO_ROOT / "wings_control" / "config"),
        help="Path to the config/ directory (default: wings_control/config/)",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also lint recipes under _experimental/ subdirectories",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    if not config_root.is_dir():
        print(f"ERROR: config root does not exist: {config_root}", file=sys.stderr)
        sys.exit(1)

    sys.exit(lint_all(config_root, args.include_experimental, args.output_json))


if __name__ == "__main__":
    main()
