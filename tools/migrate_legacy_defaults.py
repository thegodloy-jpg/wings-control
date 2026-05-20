#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equivalency regression tool for wings-control Phase D (T25 / spec §13.5).

Compares parameters produced by the legacy model_deploy_config path against
the new recipe primary path for every <engine, chip, architecture, model>
combination found in the legacy JSON files.  Outputs a migration report
showing coverage and per-field differences.

This tool does NOT rewrite any files.  It is read-only.

Usage:
  python tools/migrate_legacy_defaults.py [--config-root PATH] [--json]
  python tools/migrate_legacy_defaults.py --chip 910b-32 --engine vllm_ascend
  python tools/migrate_legacy_defaults.py --summary-only

Exit codes:
  0 – all covered combinations have zero field differences
  1 – at least one covered combination has field differences
  2 – some combinations have no matching recipe (coverage gaps)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "wings_control"))

from core.hardware_detect import KNOWN_CHIP_IDS, normalize_chip_id  # noqa: E402
from core.recipe_loader import (  # noqa: E402
    CONFIG_ROOT,
    RecipeRuntimeContext,
    _merged_recipe_defaults,
    find_recipe_candidates,
    select_best_recipe,
)

# Legacy JSON files to scan, in priority order per device type
_LEGACY_FILES = {
    "ascend": ["vllm_default.json", "ascend_default.json"],
    "nvidia": ["vllm_default.json", "nvidia_default.json"],
}

# Engine → device type mapping
_ENGINE_DEVICE = {
    "vllm_ascend": "ascend",
    "vllm_ascend_distributed": "ascend",
    "mindie": "ascend",
    "mindie_distributed": "ascend",
    "vllm": "nvidia",
    "vllm_distributed": "nvidia",
    "sglang": "nvidia",
    "sglang_distributed": "nvidia",
}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_legacy_entries(config_root: Path) -> list[dict]:
    """Yield (device, engine, architecture, model_name, params) from legacy JSONs."""
    entries = []
    seen = set()
    for device, filenames in _LEGACY_FILES.items():
        for fname in filenames:
            p = config_root / "defaults" / fname
            if not p.is_file():
                continue
            data = _load_json(p)
            mdc = data.get("model_deploy_config", {})
            for task in ("llm", "embedding", "rerank"):
                task_block = mdc.get(task, {})
                if not isinstance(task_block, dict):
                    continue
                for arch_or_global, arch_block in task_block.items():
                    if arch_or_global == "default":
                        continue
                    if not isinstance(arch_block, dict):
                        continue
                    for model_or_default, model_block in arch_block.items():
                        if model_or_default == "default":
                            continue
                        if not isinstance(model_block, dict):
                            continue
                        for engine, params in model_block.items():
                            if not isinstance(params, dict):
                                continue
                            key = (device, engine, arch_or_global, model_or_default)
                            if key not in seen:
                                seen.add(key)
                                entries.append({
                                    "device": device,
                                    "engine": engine,
                                    "architecture": arch_or_global,
                                    "model_name": model_or_default,
                                    "legacy_params": params,
                                    "source_file": fname,
                                })
    return entries


def _recipe_params(
    config_root: Path,
    engine: str,
    chip_id: str,
    architecture: str,
    model_name: str,
) -> dict | None:
    ctx = RecipeRuntimeContext(
        engine=engine,
        version="",
        chip_id=chip_id,
        model_name=model_name,
        model_path="",
        architecture=architecture,
        allow_experimental=False,
    )
    arch_c, model_c = find_recipe_candidates(ctx, config_root=config_root)
    arch_sel = select_best_recipe(arch_c)
    model_sel = select_best_recipe(model_c)
    if not arch_sel.selected and not model_sel.selected:
        return None
    return _merged_recipe_defaults(arch_sel.selected, model_sel.selected)


def _diff_params(legacy: dict, recipe: dict) -> list[dict]:
    all_keys = set(legacy) | set(recipe)
    return [
        {"field": k, "legacy": legacy.get(k), "recipe": recipe.get(k)}
        for k in sorted(all_keys)
        if legacy.get(k) != recipe.get(k)
    ]


def run_migration_report(
    config_root: Path,
    filter_chip: str = "",
    filter_engine: str = "",
    summary_only: bool = False,
    output_json: bool = False,
) -> int:
    entries = _iter_legacy_entries(config_root)
    if not entries:
        msg = "No legacy model_deploy_config entries found."
        print(json.dumps({"error": msg}) if output_json else f"WARN: {msg}", file=sys.stderr)
        return 0

    # Determine chips to evaluate for each device type
    chip_map: dict[str, list[str]] = {
        "ascend": ["910b-32", "910b-64", "910c"],
        "nvidia": ["h20-96", "h20-141", "l20"],
    }
    if filter_chip:
        chip_id, _ = normalize_chip_id(filter_chip)
        chip_map = {"ascend": [chip_id], "nvidia": [chip_id]} if chip_id else chip_map

    results = []
    total_covered = 0
    total_gaps = 0
    total_diffs = 0

    for entry in entries:
        if filter_engine and entry["engine"] != filter_engine:
            continue
        device = entry["device"]
        for chip_id in chip_map.get(device, []):
            recipe = _recipe_params(
                config_root,
                entry["engine"],
                chip_id,
                entry["architecture"],
                entry["model_name"],
            )
            if recipe is None:
                total_gaps += 1
                results.append({
                    "covered": False,
                    "engine": entry["engine"],
                    "chip": chip_id,
                    "architecture": entry["architecture"],
                    "model": entry["model_name"],
                    "source_file": entry["source_file"],
                    "diffs": [],
                })
            else:
                total_covered += 1
                diffs = _diff_params(entry["legacy_params"], recipe)
                total_diffs += len(diffs)
                results.append({
                    "covered": True,
                    "engine": entry["engine"],
                    "chip": chip_id,
                    "architecture": entry["architecture"],
                    "model": entry["model_name"],
                    "source_file": entry["source_file"],
                    "diff_count": len(diffs),
                    "diffs": diffs,
                })

    summary = {
        "total_entries": len(results),
        "covered": total_covered,
        "gaps": total_gaps,
        "total_field_diffs": total_diffs,
        "coverage_pct": round(100 * total_covered / len(results), 1) if results else 0,
    }

    if output_json:
        out = {"summary": summary}
        if not summary_only:
            out["results"] = results
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Legacy defaults migration equivalency report")
        print(f"{'='*60}")
        print(f"  Total combinations : {summary['total_entries']}")
        print(f"  Recipe covered     : {summary['covered']} ({summary['coverage_pct']}%)")
        print(f"  No recipe (gap)    : {summary['gaps']}")
        print(f"  Field differences  : {summary['total_field_diffs']}")
        print()
        if not summary_only:
            for r in results:
                tag = "OK  " if r["covered"] and r["diff_count"] == 0 else \
                      "DIFF" if r["covered"] else "GAP "
                label = f"{r['engine']}/{r['chip']}/{r['architecture']}/{r['model']}"
                if r["covered"] and r.get("diff_count", 0) > 0:
                    print(f"  {tag} {label} ({r['diff_count']} diff(s))")
                    for d in r["diffs"]:
                        print(f"       {d['field']}: legacy={d['legacy']!r}  recipe={d['recipe']!r}")
                elif not r["covered"]:
                    print(f"  {tag} {label}")
        print()

    exit_code = 0
    if total_gaps > 0:
        exit_code = 2
    if total_diffs > 0:
        exit_code = max(exit_code, 1)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equivalency regression: legacy model_deploy_config vs recipe path."
    )
    parser.add_argument(
        "--config-root",
        default=str(_REPO_ROOT / "wings_control" / "config"),
        help="Path to config/ directory",
    )
    parser.add_argument("--chip", default="", help="Filter to one chip alias (e.g. 910b-32)")
    parser.add_argument("--engine", default="", help="Filter to one engine (e.g. vllm_ascend)")
    parser.add_argument("--summary-only", action="store_true", help="Print summary counts only")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    if not config_root.is_dir():
        print(f"ERROR: config root not found: {config_root}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_migration_report(
        config_root=config_root,
        filter_chip=args.chip,
        filter_engine=args.engine,
        summary_only=args.summary_only,
        output_json=args.output_json,
    ))


if __name__ == "__main__":
    main()
