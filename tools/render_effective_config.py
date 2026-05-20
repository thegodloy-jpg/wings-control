#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the effective parameter set for a (model, engine) tuple with provenance.

Re-walks the same Phase D carriers that ``model_deploy_compat_loader`` assembles
(``deviations/<engine>.yaml`` → ``recipes/architectures/`` → ``recipes/models/``)
but, for every parameter that survives the merge, records which file produced
the value and at which merge layer.  The output lets a human read the "whole
picture" without having to mentally trace four files.

Layer priority (low → high, same as the assembler):
    global_deviation < arch_deviation < arch_recipe < model_recipe

For each canonical key the tool also shows the engine-native field name
(after the mappings translation) so the rendered output matches what the
engine command line will actually look like.

Usage:
  python tools/render_effective_config.py --model DeepSeek-R1 --engine vllm
  python tools/render_effective_config.py --architecture Qwen3MoeForCausalLM --device ascend
  python tools/render_effective_config.py --model DeepSeek-R1 --hardware h20-141 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from _config_tooling import iter_config_files, load_structured_file

_REPO_ROOT = Path(__file__).resolve().parent.parent

_ASCEND_ENGINES = ["vllm_ascend", "vllm_ascend_distributed", "mindie", "mindie_distributed"]
_NVIDIA_ENGINES = ["vllm", "vllm_distributed", "sglang", "sglang_distributed"]
_DISTRIBUTED_BASE = {
    "vllm_distributed": "vllm",
    "vllm_ascend_distributed": "vllm_ascend",
    "sglang_distributed": "sglang",
}


# ---------------------------------------------------------------------------
# Provenance-aware translation
# ---------------------------------------------------------------------------

def _engine_entry(mappings: dict[str, Any], canon_key: str, engine: str) -> dict[str, Any] | None:
    engine_map = mappings.get(canon_key)
    if not isinstance(engine_map, dict):
        return None
    entry = engine_map.get(engine) or engine_map.get(_DISTRIBUTED_BASE.get(engine, ""))
    return entry if isinstance(entry, dict) else None


def _translate_one(
    canon_key: str,
    value: Any,
    engine: str,
    mappings: dict[str, Any],
) -> tuple[str, Any] | None:
    """Translate (canon_key, value) → (native_key, native_value).

    Returns None if the engine drops the key (engine not in map, or value_map
    present but does not contain this value).
    """
    if canon_key not in mappings:
        return canon_key, value
    entry = _engine_entry(mappings, canon_key, engine)
    if entry is None:
        return None
    native_key = entry.get("field") or canon_key
    value_map = entry.get("value_map")
    if value_map is not None:
        if str(value) not in value_map:
            return None
        return native_key, value_map[str(value)]
    return native_key, value


# ---------------------------------------------------------------------------
# Carrier loading (with file paths preserved for provenance)
# ---------------------------------------------------------------------------

def _load_mappings(config_root: Path) -> dict[str, Any]:
    path = config_root / "mappings" / "canonical_to_engines.yaml"
    if not path.is_file():
        return {}
    data = load_structured_file(path)
    return data.get("canonical", {}) if isinstance(data, dict) else {}


def _load_deviation_items(
    config_root: Path, engine: str
) -> tuple[Path, list[dict[str, Any]]]:
    path = config_root / "deviations" / f"{engine}.yaml"
    if not path.is_file():
        return path, []
    data = load_structured_file(path)
    items = data.get("deviations") or []
    return path, [i for i in items if isinstance(i, dict)]


def _load_arch_recipes(config_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """arch_name → (recipe_path, recipe_data). Aliases share the same tuple."""
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in iter_config_files(config_root / "recipes" / "architectures"):
        if path.name.startswith("_"):
            continue
        data = load_structured_file(path)
        arch = data.get("architecture")
        if not arch:
            continue
        result[str(arch)] = (path, data)
        for alias in data.get("also_matches") or []:
            result[str(alias)] = (path, data)
    return result


def _load_model_recipes(
    config_root: Path,
) -> dict[str, dict[str, tuple[Path, dict[str, Any]]]]:
    """arch → model_name → (recipe_path, recipe_data)."""
    result: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    for path in iter_config_files(config_root / "recipes" / "models"):
        if path.name.startswith("_"):
            continue
        data = load_structured_file(path)
        arch = data.get("inherits_architecture")
        model_id = data.get("model_id")
        if not arch or not model_id:
            continue
        names = list((data.get("matches") or {}).get("model_names") or [model_id])
        if model_id not in names:
            names.append(model_id)
        for name in names:
            result.setdefault(str(arch), {})[str(name)] = (path, data)
    return result


# ---------------------------------------------------------------------------
# Layer extractors
# ---------------------------------------------------------------------------

def _hardware_matches(scope: list[str] | None, hardware: str | None) -> bool:
    if not scope:
        return True
    if hardware is None:
        return True
    return hardware in scope or "*" in scope


def _layer_from_global_deviations(
    config_root: Path,
    engine: str,
    mappings: dict[str, Any],
    hardware: str | None,
) -> list[dict[str, Any]]:
    """Return list of provenance records for global deviations (no arch filter)."""
    path, items = _load_deviation_items(config_root, engine)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if item.get("applies_to_architectures"):
            continue
        field = item.get("field")
        value = item.get("wings_value")
        if field is None or value is None:
            continue
        if not _hardware_matches(item.get("applies_to_hardware"), hardware):
            continue
        translated = _translate_one(str(field), value, engine, mappings)
        if translated is None:
            continue
        native_key, native_value = translated
        records.append({
            "layer": "global_deviation",
            "canonical_key": field,
            "native_key": native_key,
            "value": native_value,
            "source_file": str(path.relative_to(config_root.parent)),
            "source_index": f"deviations[{index}]",
            "applies_to_hardware": item.get("applies_to_hardware"),
        })
    return records


def _layer_from_arch_deviations(
    config_root: Path,
    arch: str,
    engine: str,
    mappings: dict[str, Any],
    hardware: str | None,
) -> list[dict[str, Any]]:
    path, items = _load_deviation_items(config_root, engine)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        archs = item.get("applies_to_architectures")
        if not archs or arch not in archs:
            continue
        field = item.get("field")
        value = item.get("wings_value")
        if field is None or value is None:
            continue
        if not _hardware_matches(item.get("applies_to_hardware"), hardware):
            continue
        translated = _translate_one(str(field), value, engine, mappings)
        if translated is None:
            continue
        native_key, native_value = translated
        records.append({
            "layer": "arch_deviation",
            "canonical_key": field,
            "native_key": native_key,
            "value": native_value,
            "source_file": str(path.relative_to(config_root.parent)),
            "source_index": f"deviations[{index}]",
            "applies_to_hardware": item.get("applies_to_hardware"),
        })
    return records


def _layer_from_arch_recipe(
    arch_path: Path,
    arch_data: dict[str, Any],
    engine: str,
    mappings: dict[str, Any],
    config_root: Path,
) -> list[dict[str, Any]]:
    defaults = arch_data.get("defaults") or {}
    if not isinstance(defaults, dict):
        return []
    records: list[dict[str, Any]] = []
    for canon_key, value in defaults.items():
        translated = _translate_one(str(canon_key), value, engine, mappings)
        if translated is None:
            continue
        native_key, native_value = translated
        records.append({
            "layer": "arch_recipe",
            "canonical_key": canon_key,
            "native_key": native_key,
            "value": native_value,
            "source_file": str(arch_path.relative_to(config_root.parent)),
            "source_index": "defaults",
            "applies_to_hardware": None,
        })
    return records


def _layer_from_model_recipe(
    model_path: Path,
    model_data: dict[str, Any],
    engine: str,
    mappings: dict[str, Any],
    config_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (apply_records, null_records).

    null_records have ``value`` set to None and signal that the model recipe
    explicitly cancels an inherited canonical key.
    """
    defaults = model_data.get("defaults") or {}
    if not isinstance(defaults, dict):
        return [], []
    apply: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    for canon_key, value in defaults.items():
        if value is None:
            # cancellation — needs native_key lookup for the right pop()
            entry = _engine_entry(mappings, str(canon_key), engine)
            native_key = (entry.get("field") if entry else None) or canon_key
            nulls.append({
                "layer": "model_recipe",
                "canonical_key": canon_key,
                "native_key": native_key,
                "value": None,
                "source_file": str(model_path.relative_to(config_root.parent)),
                "source_index": "defaults",
                "applies_to_hardware": None,
                "cancels_inherited": True,
            })
            continue
        translated = _translate_one(str(canon_key), value, engine, mappings)
        if translated is None:
            continue
        native_key, native_value = translated
        apply.append({
            "layer": "model_recipe",
            "canonical_key": canon_key,
            "native_key": native_key,
            "value": native_value,
            "source_file": str(model_path.relative_to(config_root.parent)),
            "source_index": "defaults",
            "applies_to_hardware": None,
        })
    return apply, nulls


# ---------------------------------------------------------------------------
# Effective merge
# ---------------------------------------------------------------------------

def _merge_layered_records(
    layers: list[list[dict[str, Any]]],
    null_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Merge ordered layers; later layers overwrite earlier.  Returns:
       - effective: native_key → winning record (with ``overridden_by`` history)
       - history: full ordered list of every applied record (for the trace)
    """
    effective: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for records in layers:
        for record in records:
            history.append(record)
            previous = effective.get(record["native_key"])
            if previous is not None:
                record = dict(record)
                record["overrides"] = {
                    "previous_value": previous["value"],
                    "previous_layer": previous["layer"],
                    "previous_source": previous["source_file"],
                }
            effective[record["native_key"]] = record
    for null_record in null_records:
        history.append(null_record)
        effective.pop(null_record["native_key"], None)
    return effective, history


def _render_one(
    config_root: Path,
    arch: str,
    engine: str,
    model_name: str | None,
    model_path: Path | None,
    model_data: dict[str, Any] | None,
    arch_recipes: dict[str, tuple[Path, dict[str, Any]]],
    mappings: dict[str, Any],
    hardware: str | None,
) -> dict[str, Any]:
    arch_path_data = arch_recipes.get(arch)
    arch_path = arch_path_data[0] if arch_path_data else None
    arch_data = arch_path_data[1] if arch_path_data else {}

    layer_globals = _layer_from_global_deviations(config_root, engine, mappings, hardware)
    layer_arch_dev = _layer_from_arch_deviations(config_root, arch, engine, mappings, hardware)
    layer_arch_recipe: list[dict[str, Any]] = []
    if arch_path is not None:
        layer_arch_recipe = _layer_from_arch_recipe(
            arch_path, arch_data, engine, mappings, config_root
        )
    layer_model_apply: list[dict[str, Any]] = []
    layer_model_nulls: list[dict[str, Any]] = []
    if model_path is not None and model_data is not None:
        layer_model_apply, layer_model_nulls = _layer_from_model_recipe(
            model_path, model_data, engine, mappings, config_root
        )

    effective, history = _merge_layered_records(
        [layer_globals, layer_arch_dev, layer_arch_recipe, layer_model_apply],
        layer_model_nulls,
    )

    sources: list[str] = []
    for record in history:
        src = record["source_file"]
        if src not in sources:
            sources.append(src)

    return {
        "architecture": arch,
        "engine": engine,
        "model": model_name,
        "hardware": hardware,
        "effective": {key: deepcopy(record) for key, record in sorted(effective.items())},
        "trace": history,
        "sources_read": sources,
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _resolve_engines(device: str | None, engine: str | None) -> list[str]:
    if engine:
        return [engine]
    if device == "ascend":
        return list(_ASCEND_ENGINES)
    if device == "nvidia":
        return list(_NVIDIA_ENGINES)
    return list(_NVIDIA_ENGINES) + list(_ASCEND_ENGINES)


def _resolve_targets(
    config_root: Path,
    architecture: str | None,
    model: str | None,
) -> list[tuple[str, str | None, Path | None, dict[str, Any] | None]]:
    """Return (arch, model_name, model_path, model_data). model fields None when whole arch requested."""
    arch_recipes = _load_arch_recipes(config_root)
    model_recipes = _load_model_recipes(config_root)

    if model:
        for arch, models in model_recipes.items():
            for name, (path, data) in models.items():
                if name == model:
                    return [(arch, name, path, data)]
        raise SystemExit(f"ERROR: model {model!r} not found in recipes/models/")

    if architecture:
        if architecture not in arch_recipes and architecture not in model_recipes:
            raise SystemExit(
                f"ERROR: architecture {architecture!r} not found in recipes/architectures/"
            )
        return [(architecture, None, None, None)]

    targets: list[tuple[str, str | None, Path | None, dict[str, Any] | None]] = []
    for arch in sorted(arch_recipes):
        targets.append((arch, None, None, None))
    return targets


def _format_text(reports: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for report in reports:
        head = f"{report['architecture']} / {report['model'] or '<arch default>'} / {report['engine']}"
        if report.get("hardware"):
            head += f"  (hardware={report['hardware']})"
        lines.append("=" * len(head))
        lines.append(head)
        lines.append("=" * len(head))
        effective = report["effective"]
        if not effective:
            lines.append("  (no parameters)")
        else:
            key_width = max(len(k) for k in effective) + 2
            for native_key, record in effective.items():
                value_repr = json.dumps(record["value"], ensure_ascii=False)
                canon = record["canonical_key"]
                origin = f"{record['source_file']}#{record['source_index']}"
                tag = "" if canon == native_key else f"  (canonical={canon})"
                lines.append(
                    f"  {native_key.ljust(key_width)}= {value_repr}"
                    f"  <- {record['layer']}  [{origin}]{tag}"
                )
                if "overrides" in record:
                    prev = record["overrides"]
                    lines.append(
                        f"  {' ' * key_width}  overrides "
                        f"{json.dumps(prev['previous_value'], ensure_ascii=False)} "
                        f"from {prev['previous_layer']} [{prev['previous_source']}]"
                    )
        nulls = [r for r in report["trace"] if r.get("cancels_inherited")]
        for r in nulls:
            lines.append(
                f"  -- cancelled {r['native_key']} (canonical={r['canonical_key']}) "
                f"via {r['source_file']}#{r['source_index']}"
            )
        lines.append("")
        lines.append("Sources read:")
        for src in report["sources_read"]:
            lines.append(f"  - {src}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the effective parameter set for a (model, engine) tuple with provenance."
    )
    parser.add_argument("--config-root", default=str(_REPO_ROOT / "wings_control" / "config"))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--model", help="Model recipe name (matches recipes/models/*.model_id or matches.model_names).")
    target.add_argument("--architecture", help="Architecture name (matches recipes/architectures/*.architecture).")
    parser.add_argument("--engine", help="Restrict output to this engine (defaults to all engines for the device).")
    parser.add_argument("--device", choices=["nvidia", "ascend"], help="Limit engines to this device family.")
    parser.add_argument("--hardware", help="Chip id; deviations whose applies_to_hardware excludes this are filtered out.")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Emit JSON instead of the text view.")
    args = parser.parse_args()

    config_root = Path(args.config_root).resolve()
    if not config_root.is_dir():
        print(f"ERROR: config root does not exist: {config_root}", file=sys.stderr)
        sys.exit(1)

    engines = _resolve_engines(args.device, args.engine)
    targets = _resolve_targets(config_root, args.architecture, args.model)
    mappings = _load_mappings(config_root)
    arch_recipes = _load_arch_recipes(config_root)

    reports: list[dict[str, Any]] = []
    for arch, model_name, model_path, model_data in targets:
        for engine in engines:
            reports.append(
                _render_one(
                    config_root,
                    arch,
                    engine,
                    model_name,
                    model_path,
                    model_data,
                    arch_recipes,
                    mappings,
                    args.hardware,
                )
            )

    if args.output_json:
        print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2, default=str))
    else:
        print(_format_text(reports), end="")


if __name__ == "__main__":
    main()
