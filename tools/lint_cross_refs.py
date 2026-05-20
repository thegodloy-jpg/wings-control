#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-file reference lint for Phase D carriers.

The per-file linters (``lint_deviations.py``, ``lint_mappings.py``,
``lint_recipes.py``) check intra-file shape but do not catch dangling
references between carriers.  This tool fills that gap.

Checks performed:

  1.  ``recipes/models/<m>.yaml`` ``inherits_architecture`` resolves to an
      architecture (or ``also_matches`` alias) actually defined under
      ``recipes/architectures/``.
  2.  ``deviations/<engine>.yaml`` ``applies_to_architectures`` entries
      resolve to a known architecture.
  3.  No ``matches.model_names`` collisions across model recipes (two recipes
      claiming the same model_name is ambiguous for the loader).
  4.  Mapping cross-checks:
        - Deviation ``field`` not present in mappings is reported as a warning
          (the assembler will pass it through unchanged, which is rarely what
          a non-canonical-typo author intended).
        - Canonical keys defined in mappings but referenced by *no*
          deviation / arch recipe / model recipe are flagged as orphans.
        - Architecture / model defaults referencing a canonical key that
          *is* present in mappings but lists no engine entry for any engine
          on either device family.

Exit codes:
  0 – clean (warnings may still print)
  1 – at least one error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _config_tooling import (
    SUPPORTED_ENGINES,
    iter_config_files,
    load_structured_file,
    print_results,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Carrier collectors
# ---------------------------------------------------------------------------

def _collect_architectures(config_root: Path) -> dict[str, Path]:
    """Return {arch_name: source_path}; also_matches aliases included."""
    result: dict[str, Path] = {}
    for path in iter_config_files(config_root / "recipes" / "architectures"):
        if path.name.startswith("_"):
            continue
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        arch = data.get("architecture")
        if arch:
            result[str(arch)] = path
        for alias in data.get("also_matches") or []:
            result[str(alias)] = path
    return result


def _collect_models(
    config_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in iter_config_files(config_root / "recipes" / "models"):
        if path.name.startswith("_"):
            continue
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        items.append((path, data))
    return items


def _collect_deviations(
    config_root: Path,
) -> list[tuple[Path, str, list[dict[str, Any]]]]:
    items: list[tuple[Path, str, list[dict[str, Any]]]] = []
    for path in iter_config_files(config_root / "deviations"):
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        engine = str(data.get("engine") or path.stem)
        devs = data.get("deviations") or []
        if isinstance(devs, list):
            items.append((path, engine, [d for d in devs if isinstance(d, dict)]))
    return items


def _collect_mappings(config_root: Path) -> tuple[Path | None, dict[str, Any]]:
    path = config_root / "mappings" / "canonical_to_engines.yaml"
    if not path.is_file():
        return None, {}
    try:
        data = load_structured_file(path)
    except Exception:
        return path, {}
    return path, data.get("canonical", {}) if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.parent))
    except ValueError:
        return str(path)


def _check_model_inherits(
    config_root: Path,
    archs: dict[str, Path],
    models: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path, data in models:
        diagnostics: list[tuple[str, str]] = []
        inherits = data.get("inherits_architecture")
        if not inherits:
            diagnostics.append(("error", "missing inherits_architecture"))
        elif str(inherits) not in archs:
            diagnostics.append((
                "error",
                f"inherits_architecture {inherits!r} not defined in recipes/architectures/",
            ))
        if diagnostics:
            results.append({
                "file": _rel(path, config_root),
                "diagnostics": [{"level": lvl, "message": msg} for lvl, msg in diagnostics],
            })
    return results


def _check_deviation_arch_refs(
    config_root: Path,
    archs: dict[str, Path],
    deviations: list[tuple[Path, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path, _engine, items in deviations:
        diagnostics: list[tuple[str, str]] = []
        for index, item in enumerate(items):
            arch_list = item.get("applies_to_architectures")
            if not arch_list:
                continue
            if not isinstance(arch_list, list):
                diagnostics.append((
                    "error",
                    f"deviations[{index}].applies_to_architectures must be a list",
                ))
                continue
            for arch in arch_list:
                if str(arch) not in archs:
                    diagnostics.append((
                        "error",
                        f"deviations[{index}].applies_to_architectures references "
                        f"unknown architecture: {arch!r}",
                    ))
        if diagnostics:
            results.append({
                "file": _rel(path, config_root),
                "diagnostics": [{"level": lvl, "message": msg} for lvl, msg in diagnostics],
            })
    return results


def _check_duplicate_model_names(
    config_root: Path,
    models: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen: dict[str, list[Path]] = {}
    for path, data in models:
        model_id = data.get("model_id")
        names: set[str] = set()
        matches = data.get("matches") or {}
        for name in matches.get("model_names") or []:
            names.add(str(name))
        if model_id:
            names.add(str(model_id))
        for name in names:
            seen.setdefault(name, []).append(path)

    results: list[dict[str, Any]] = []
    for name, paths in seen.items():
        if len(paths) <= 1:
            continue
        owners = ", ".join(_rel(p, config_root) for p in paths)
        for path in paths:
            results.append({
                "file": _rel(path, config_root),
                "diagnostics": [{
                    "level": "error",
                    "message": f"model_name {name!r} also claimed by: {owners}",
                }],
            })
    return results


def _check_mapping_orphans_and_dev_keys(
    config_root: Path,
    mapping_path: Path | None,
    mappings: dict[str, Any],
    deviations: list[tuple[Path, str, list[dict[str, Any]]]],
    arch_files: list[tuple[Path, dict[str, Any]]],
    models: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    used_canon: set[str] = set()
    results: list[dict[str, Any]] = []

    # Deviations: field MUST appear in mappings (otherwise it pass-throughs as
    # an engine-private key under whatever the author wrote — a sharp edge).
    for path, _engine, items in deviations:
        diagnostics: list[tuple[str, str]] = []
        for index, item in enumerate(items):
            field = item.get("field")
            if not field:
                continue
            used_canon.add(str(field))
            if str(field) not in mappings:
                diagnostics.append((
                    "warning",
                    f"deviations[{index}].field {field!r} has no entry in "
                    f"mappings/canonical_to_engines.yaml (will pass through unchanged)",
                ))
        if diagnostics:
            results.append({
                "file": _rel(path, config_root),
                "diagnostics": [{"level": lvl, "message": msg} for lvl, msg in diagnostics],
            })

    # Recipe defaults: canonical keys *may* be unmapped (pass-through is fine
    # for keys all engines share), but track usage for orphan detection.
    for _path, data in arch_files:
        for key in (data.get("defaults") or {}).keys():
            used_canon.add(str(key))
    for _path, data in models:
        for key in (data.get("defaults") or {}).keys():
            used_canon.add(str(key))

    # Orphans: keys in mappings unused everywhere.  We exclude well-known
    # runtime-injected keys that the assembler never sees in recipes (host,
    # port, model_name, model_path) — these come from wings_entry at deploy
    # time, not from carriers.
    runtime_keys = {"host", "port", "model_name", "model_path"}
    if mapping_path is not None:
        orphans = sorted(
            key for key in mappings.keys() if key not in used_canon and key not in runtime_keys
        )
        if orphans:
            results.append({
                "file": _rel(mapping_path, config_root),
                "diagnostics": [{
                    "level": "warning",
                    "message": (
                        f"{len(orphans)} canonical key(s) defined but unused by any "
                        f"deviation or recipe: {', '.join(orphans)}"
                    ),
                }],
            })

    # Mapping entries that list zero engine targets — already a warning in
    # lint_mappings, but worth flagging cross-file too when the key IS used:
    if mapping_path is not None:
        empty_used: list[str] = []
        for key in sorted(used_canon):
            entry = mappings.get(key)
            if not isinstance(entry, dict):
                continue
            engines = [e for e in entry if e in SUPPORTED_ENGINES]
            if not engines:
                empty_used.append(key)
        if empty_used:
            results.append({
                "file": _rel(mapping_path, config_root),
                "diagnostics": [{
                    "level": "error",
                    "message": (
                        f"canonical key(s) referenced by carriers but mapped to no "
                        f"supported engine: {', '.join(empty_used)}"
                    ),
                }],
            })

    return results


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def lint_all(config_root: Path, output_json: bool) -> int:
    archs = _collect_architectures(config_root)
    models = _collect_models(config_root)
    deviations = _collect_deviations(config_root)
    mapping_path, mappings = _collect_mappings(config_root)

    results: list[dict[str, Any]] = []
    results.extend(_check_model_inherits(config_root, archs, models))
    results.extend(_check_deviation_arch_refs(config_root, archs, deviations))
    results.extend(_check_duplicate_model_names(config_root, models))
    results.extend(
        _check_mapping_orphans_and_dev_keys(
            config_root, mapping_path, mappings, deviations, list(_iter_arch_files(config_root)), models
        )
    )
    return print_results(results, output_json=output_json)


def _iter_arch_files(config_root: Path):
    for path in iter_config_files(config_root / "recipes" / "architectures"):
        if path.name.startswith("_"):
            continue
        try:
            yield path, load_structured_file(path)
        except Exception:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lint cross-file references between Phase D carriers."
    )
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
