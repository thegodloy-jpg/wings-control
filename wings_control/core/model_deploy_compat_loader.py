# -*- coding: utf-8 -*-
"""Phase D parameter assembler — replaces the static model_deploy_compat carrier.

Assembles the legacy ``model_deploy_config`` structure expected by
``config_loader._load_and_validate_models_dict`` from the proper Phase D
carriers:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  deviations/<engine>.yaml   → global defaults (all architectures)       │
  │  deviations/<engine>.yaml   → arch-specific defaults (applies_to_arches)│
  │  recipes/architectures/     → canonical arch-level params               │
  │  recipes/models/            → canonical model-level overrides           │
  │  mappings/canonical_to_engines.yaml → key + value translation           │
  └─────────────────────────────────────────────────────────────────────────┘

Merge priority (low → high):
  global_deviation < arch_deviation < arch_recipe < model_recipe

Null values in a model recipe explicitly cancel an inherited key (e.g.
``reasoning_parser: null`` removes reasoning_parser set by the arch recipe).

Value translation:
  Canonical defaults use vllm-format values (e.g. tool_call_parser=deepseek_v3).
  For engines with different naming conventions (sglang, mindie) the mappings
  ``value_map`` entry translates values to engine-native format.  If an engine
  has a value_map and the canonical value is NOT in the map, the key is DROPPED
  for that engine (the engine doesn't support that parser/value).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"

_ASCEND_ENGINES = ["vllm_ascend", "vllm_ascend_distributed", "mindie", "mindie_distributed"]
_NVIDIA_ENGINES = ["vllm", "vllm_distributed", "sglang", "sglang_distributed"]

# Distributed engines that inherit mappings from their base engine.
_DISTRIBUTED_BASE = {
    "vllm_distributed": "vllm",
    "vllm_ascend_distributed": "vllm_ascend",
    "sglang_distributed": "sglang",
}

# Chips treated as belonging to each device family when no explicit chip is
# pinned in hardware_env.  Matches the canonical chip ids declared by
# hardware_profiles/*.yaml.
_DEVICE_CHIPS = {
    "ascend": ["910b-32", "910b-64", "910c"],
    "nvidia": ["h20-96", "h20-141", "l20", "rtx-pro-5000"],
}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            return {}
        data = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not load %s: %s", path, exc)
        return {}


def _iter_yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        f for pat in ("*.yaml", "*.yml", "*.json")
        for f in directory.glob(pat)
        if f.is_file() and not f.name.startswith("_")
    )


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

def _load_mappings(config_root: Path) -> dict[str, Any]:
    path = config_root / "mappings" / "canonical_to_engines.yaml"
    data = _load_file(path)
    return data.get("canonical", {}) if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Parameter translation
# ---------------------------------------------------------------------------

def _translate_for_engine(
    canonical_params: dict[str, Any],
    engine: str,
    mappings: dict[str, Any],
) -> dict[str, Any]:
    """Translate canonical key→value pairs to engine-native form.

    Rules:
    - Key in mappings with engine entry: rename key; apply value_map if present.
    - Key in mappings but engine NOT listed: DROP the key (engine doesn't use it).
    - Key NOT in mappings at all: pass through unchanged.
    - If engine has value_map and canonical value is NOT in map: DROP the key.
    """
    result: dict[str, Any] = {}
    for canon_key, value in canonical_params.items():
        if canon_key not in mappings:
            result[canon_key] = value
            continue
        engine_map = mappings[canon_key]
        if not isinstance(engine_map, dict):
            result[canon_key] = value
            continue
        entry = engine_map.get(engine) or engine_map.get(_DISTRIBUTED_BASE.get(engine, ""))
        if entry is None:
            continue  # engine not listed → drop
        native_key = entry.get("field") or canon_key
        value_map = entry.get("value_map")
        if value_map is not None:
            if str(value) not in value_map:
                continue  # engine doesn't support this value → drop
            native_value = value_map[str(value)]
        else:
            native_value = value
        result[native_key] = native_value
    return result


def _get_native_key(canon_key: str, engine: str, mappings: dict[str, Any]) -> str:
    """Return the engine-native field name for a canonical key."""
    engine_map = mappings.get(canon_key, {})
    entry = engine_map.get(engine) or engine_map.get(_DISTRIBUTED_BASE.get(engine, ""))
    return entry.get("field") or canon_key if isinstance(entry, dict) else canon_key


# ---------------------------------------------------------------------------
# Deviations readers
# ---------------------------------------------------------------------------

def _global_deviation_params(
    config_root: Path, engine: str, mappings: dict[str, Any]
) -> dict[str, Any]:
    """Return global (non-architecture-specific) deviation params for engine."""
    path = config_root / "deviations" / f"{engine}.yaml"
    data = _load_file(path)
    result: dict[str, Any] = {}
    for item in data.get("deviations", []):
        if not isinstance(item, dict):
            continue
        if item.get("applies_to_architectures"):
            continue  # skip arch-specific entries
        field = item.get("field")
        value = item.get("wings_value")
        if field is None or value is None:
            continue
        translated = _translate_for_engine({field: value}, engine, mappings)
        result.update(translated)
    return result


def _arch_specific_deviation_params(
    config_root: Path, engines: list[str], mappings: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return arch → engine → translated params for deviations with applies_to_architectures."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in engines:
        path = config_root / "deviations" / f"{engine}.yaml"
        data = _load_file(path)
        for item in data.get("deviations", []):
            if not isinstance(item, dict):
                continue
            archs = item.get("applies_to_architectures")
            if not archs:
                continue
            field = item.get("field")
            value = item.get("wings_value")
            if field is None or value is None:
                continue
            translated = _translate_for_engine({field: value}, engine, mappings)
            if not translated:
                continue
            for arch in archs:
                result.setdefault(arch, {}).setdefault(engine, {}).update(translated)
    return result


# ---------------------------------------------------------------------------
# Recipe readers
# ---------------------------------------------------------------------------

def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — overlay wins, nested dicts are merged not replaced."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _apply_hardware_overlays(
    defaults: dict[str, Any],
    overlays: list[dict[str, Any]],
    chips: list[str],
) -> dict[str, Any]:
    """Merge any hardware_overlay whose applies_to_hardware intersects ``chips``.

    Used by the compat assembler so the resulting model_deploy_config carries
    the same hardware-specific defaults the legacy ``<device>_default.json``
    files used to encode statically per arch.
    """
    if not isinstance(overlays, list) or not chips:
        return defaults
    chip_set = set(chips)
    merged = deepcopy(defaults)
    for overlay in overlays:
        if not isinstance(overlay, dict):
            continue
        applies = overlay.get("applies_to_hardware") or []
        if not chip_set.intersection(applies):
            continue
        overlay_defaults = overlay.get("defaults") or {}
        if isinstance(overlay_defaults, dict):
            merged = _deep_merge_dict(merged, overlay_defaults)
    return merged


def _arch_canonical_defaults(
    config_root: Path,
    chips: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return arch_name → canonical defaults from recipes/architectures/.

    When ``chips`` is provided, any ``hardware_overlays`` entry matching one of
    those chips is merged on top of the base ``defaults``.
    """
    arch_dir = config_root / "recipes" / "architectures"
    result: dict[str, dict[str, Any]] = {}
    chip_list = list(chips or [])
    for path in _iter_yaml_files(arch_dir):
        data = _load_file(path)
        arch = data.get("architecture")
        if not arch:
            continue
        defaults = data.get("defaults") or {}
        if not isinstance(defaults, dict):
            continue
        merged = _apply_hardware_overlays(
            defaults, data.get("hardware_overlays") or [], chip_list
        )
        result[str(arch)] = merged
        for alias in data.get("also_matches") or []:
            result[str(alias)] = deepcopy(merged)
    return result


def _model_canonical_defaults(config_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return arch_name → model_name → canonical defaults from recipes/models/."""
    models_dir = config_root / "recipes" / "models"
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for path in _iter_yaml_files(models_dir):
        data = _load_file(path)
        arch = data.get("inherits_architecture")
        model_id = data.get("model_id")
        if not arch or not model_id:
            continue
        defaults = dict(data.get("defaults") or {})
        model_names = list(data.get("matches", {}).get("model_names") or [model_id])
        if model_id not in model_names:
            model_names.append(model_id)
        for name in model_names:
            result.setdefault(str(arch), {})[str(name)] = deepcopy(defaults)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _device_key(hardware_env: dict[str, Any] | str) -> str:
    if isinstance(hardware_env, dict):
        device = str(hardware_env.get("device") or "nvidia").strip().lower()
    else:
        device = str(hardware_env or "nvidia").strip().lower()
    if device not in {"nvidia", "ascend"}:
        logger.warning("Unsupported device type '%s', fallback to 'nvidia'", device)
        return "nvidia"
    return device


def load_migrated_model_deploy_config(
    hardware_env: dict[str, Any] | str,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Assemble model_deploy_config from Phase D carriers.

    Merge priority for each final per-engine param set:
      global_deviation < arch_deviation < arch_recipe < model_recipe

    Model entries inherit the full arch-level param set; model-level
    ``field: null`` removes a key inherited from the arch level.

    Returns ``(model_deploy_config, "phase-d-carriers")`` or ``(None, None)``.
    """
    try:
        config_root = Path(config_root)
        device = _device_key(hardware_env)
        engines = _ASCEND_ENGINES if device == "ascend" else _NVIDIA_ENGINES

        mappings = _load_mappings(config_root)

        # --- Per-engine global defaults (from deviations, no arch filter) ---
        global_by_engine: dict[str, dict[str, Any]] = {
            e: _global_deviation_params(config_root, e, mappings) for e in engines
        }

        # --- Arch-specific deviation params (keyed by architecture name) ---
        arch_dev_by_arch_engine = _arch_specific_deviation_params(
            config_root, engines, mappings
        )

        # --- Recipe defaults (with hardware overlays merged for the device) ---
        chips = _DEVICE_CHIPS.get(device, [])
        if isinstance(hardware_env, dict):
            explicit_chip = hardware_env.get("chip_id") or hardware_env.get("chip")
            if explicit_chip:
                chips = [str(explicit_chip)]
        arch_recipe = _arch_canonical_defaults(config_root, chips=chips)
        model_recipe = _model_canonical_defaults(config_root)

        # --- Build arch-level param sets (global + arch_dev + arch_recipe) ---
        # arch_engine_params[arch][engine] = full merged param dict
        def _build_arch_params(arch: str) -> dict[str, dict[str, Any]]:
            canon = arch_recipe.get(arch, {})
            result: dict[str, dict[str, Any]] = {}
            for engine in engines:
                base: dict[str, Any] = {}
                base.update(global_by_engine.get(engine, {}))
                base.update(arch_dev_by_arch_engine.get(arch, {}).get(engine, {}))
                base.update(_translate_for_engine(canon, engine, mappings))
                if base:
                    result[engine] = base
            return result

        # --- Assemble ---
        llm_global_default = {e: p for e, p in global_by_engine.items() if p}
        llm: dict[str, Any] = {"default": llm_global_default}

        all_arches = set(arch_recipe) | set(arch_dev_by_arch_engine) | set(model_recipe)
        for arch in all_arches:
            arch_params = _build_arch_params(arch)
            arch_section: dict[str, Any] = {}
            if arch_params:
                arch_section["default"] = arch_params

            for model_name, model_canon in model_recipe.get(arch, {}).items():
                model_engine_params: dict[str, dict[str, Any]] = {}
                for engine in engines:
                    base = dict(arch_params.get(engine, {}))
                    # Apply model overrides (null = remove key)
                    for canon_key, value in model_canon.items():
                        if value is None:
                            native_key = _get_native_key(canon_key, engine, mappings)
                            base.pop(native_key, None)
                        else:
                            base.update(
                                _translate_for_engine({canon_key: value}, engine, mappings)
                            )
                    if base:
                        model_engine_params[engine] = base
                if model_engine_params:
                    arch_section[model_name] = model_engine_params

            if arch_section:
                llm[arch] = arch_section

        model_deploy_config = {"llm": llm}
        logger.info(
            "Assembled model_deploy_config from Phase D carriers "
            "(device=%s, arches=%d, models=%d)",
            device,
            len(all_arches),
            sum(len(v) for v in model_recipe.values()),
        )
        return model_deploy_config, "phase-d-carriers"

    except Exception as exc:
        logger.warning(
            "Failed to assemble Phase D model_deploy_config: %s", exc, exc_info=True
        )
        return None, None


def extract_model_deploy_config(default_config: dict[str, Any]) -> dict[str, Any]:
    """Extract a validated model_deploy_config object from a legacy default file."""
    model_deploy_config = default_config.get("model_deploy_config")
    if not isinstance(model_deploy_config, dict):
        raise ValueError("legacy default config missing model_deploy_config object")
    return deepcopy(model_deploy_config)


def build_compat_document(
    device: str,
    model_deploy_config: dict[str, Any],
    *,
    source_ref: str = "",
) -> dict[str, Any]:
    """Build a compatibility carrier document (legacy helper, kept for callers)."""
    return {
        "schema_version": "model_deploy_compat.v1",
        "device": _device_key(device),
        "source_ref": source_ref,
        "migration_note": "Assembled from Phase D carriers.",
        "model_deploy_config": deepcopy(model_deploy_config),
    }
