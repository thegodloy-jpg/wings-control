#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline check tool for wings-control (Phase B T21).

Compares the engine parameters produced by the recipe primary path against the
legacy model_deploy_config path for a given <engine, chip, model> tuple, and
reports any field-level differences.

Usage:
  python tools/check_baseline.py \\
      --engine vllm_ascend \\
      --chip 910b-32 \\
      --model-name Qwen3-32B \\
      --model-path /weights/Qwen3-32B \\
      [--engine-version 0.11.0] \\
      [--allow-experimental] \\
      [--config-root PATH] \\
      [--json]

Exit codes:
  0 – zero field differences between recipe path and legacy path
  1 – differences found or error
  2 – recipe did not match (no recipe for this combination)
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
    RecipeRuntimeContext,
    find_recipe_candidates,
    read_model_architecture,
    select_best_recipe,
    _merged_recipe_defaults,
    CONFIG_ROOT,
)

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def _load_legacy_defaults(
    config_root: Path,
    engine: str,
    chip_device: str,
    architecture: str,
    model_name: str,
) -> dict:
    """Load the legacy model_deploy_config entry for this combination."""
    device = "ascend" if "ascend" in engine or chip_device == "ascend" else "nvidia"
    candidates = []
    for fname in ("vllm_default.json", f"{device}_default.json"):
        p = config_root / "defaults" / fname
        if p.is_file():
            candidates.append(p)
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception:
            continue
        mdc = data.get("model_deploy_config", {}).get("llm", {})
        arch_block = mdc.get(architecture, {})
        model_block = arch_block.get(model_name, {})
        default_block = arch_block.get("default", {})
        global_default = mdc.get("default", {})

        # Priority: model > arch.default > global default — pick engine key
        merged: dict = {}
        for block in (global_default, default_block, model_block):
            entry = block.get(engine) or block.get(engine.replace("_distributed", "")) or {}
            if isinstance(entry, dict):
                merged.update(entry)
        if merged:
            return merged
    return {}


def _diff(legacy: dict, recipe: dict) -> list[dict]:
    all_keys = set(legacy) | set(recipe)
    diffs = []
    for k in sorted(all_keys):
        lv = legacy.get(k)
        rv = recipe.get(k)
        if lv != rv:
            diffs.append({"field": k, "legacy": lv, "recipe": rv})
    return diffs


def run_check(
    engine: str,
    chip: str,
    model_name: str,
    model_path: str,
    engine_version: str,
    allow_experimental: bool,
    config_root: Path,
    output_json: bool,
) -> int:
    chip_id, _ = normalize_chip_id(chip)
    if not chip_id:
        msg = f"Unknown chip alias: {chip!r}. Known: {sorted(KNOWN_CHIP_IDS)}"
        if output_json:
            print(json.dumps({"error": msg}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    architecture = read_model_architecture(model_path)
    ctx = RecipeRuntimeContext(
        engine=engine,
        version=engine_version,
        chip_id=chip_id,
        model_name=model_name,
        model_path=model_path,
        architecture=architecture,
        allow_experimental=allow_experimental,
    )

    arch_candidates, model_candidates = find_recipe_candidates(ctx, config_root=config_root)
    arch_sel = select_best_recipe(arch_candidates)
    model_sel = select_best_recipe(model_candidates)

    if not arch_sel.selected and not model_sel.selected:
        msg = f"No recipe matched for engine={engine} chip={chip_id} model={model_name} arch={architecture!r}"
        if output_json:
            print(json.dumps({"matched": False, "message": msg}))
        else:
            print(f"WARN: {msg}", file=sys.stderr)
        return 2

    recipe_defaults = _merged_recipe_defaults(arch_sel.selected, model_sel.selected)
    legacy_defaults = _load_legacy_defaults(
        config_root, engine, "ascend" if "ascend" in engine else "nvidia",
        architecture, model_name,
    )

    diffs = _diff(legacy_defaults, recipe_defaults)
    result = {
        "matched": True,
        "engine": engine,
        "chip_id": chip_id,
        "model_name": model_name,
        "architecture": architecture,
        "recipe_arch": arch_sel.selected.path if arch_sel.selected else None,
        "recipe_model": model_sel.selected.path if model_sel.selected else None,
        "diff_count": len(diffs),
        "diffs": diffs,
        "legacy_keys": sorted(legacy_defaults),
        "recipe_keys": sorted(recipe_defaults),
    }

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Engine  : {engine}")
        print(f"Chip    : {chip_id}")
        print(f"Model   : {model_name}")
        print(f"Arch    : {architecture or '(not detected)'}")
        print(f"Matched : arch={arch_sel.selected.path if arch_sel.selected else 'none'}")
        print(f"          model={model_sel.selected.path if model_sel.selected else 'none'}")
        print()
        if not diffs:
            print("OK — 0 field differences between recipe and legacy paths.")
        else:
            print(f"DIFF — {len(diffs)} field(s) differ:")
            for d in diffs:
                print(f"  {d['field']}: legacy={d['legacy']!r}  recipe={d['recipe']!r}")

    return 1 if diffs else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recipe vs legacy defaults for a given deployment tuple.")
    parser.add_argument("--engine", required=True, help="Engine name (e.g. vllm_ascend, vllm, sglang)")
    parser.add_argument("--chip", required=True, help="Chip canonical ID or alias (e.g. 910b-32, h20-141)")
    parser.add_argument("--model-name", required=True, help="Model name as passed to --model-name")
    parser.add_argument("--model-path", default="", help="Model weight path (used to detect architecture)")
    parser.add_argument("--engine-version", default="", help="Engine version string (e.g. 0.11.0)")
    parser.add_argument("--allow-experimental", action="store_true", help="Include _experimental/ recipes")
    parser.add_argument(
        "--config-root",
        default=str(_REPO_ROOT / "wings_control" / "config"),
        help="Path to config/ directory",
    )
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    if not config_root.is_dir():
        print(f"ERROR: config root not found: {config_root}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_check(
        engine=args.engine,
        chip=args.chip,
        model_name=args.model_name,
        model_path=args.model_path,
        engine_version=args.engine_version,
        allow_experimental=args.allow_experimental,
        config_root=config_root,
        output_json=args.output_json,
    ))


if __name__ == "__main__":
    main()
