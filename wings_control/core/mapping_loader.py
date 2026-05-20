# -*- coding: utf-8 -*-
"""Phase D canonical-to-engine parameter mapping loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging

try:  # pragma: no cover - deployment images may include PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover - bootstrap environments may not
    yaml = None

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
CANONICAL_MAPPING_FILE = "canonical_to_engines.yaml"
COMMON_CANONICAL_KEYS = (
    "host",
    "port",
    "model_name",
    "model_path",
    "input_length",
    "output_length",
    "trust_remote_code",
    "dtype",
    "kv_cache_dtype",
    "quantization",
    "quantization_param_path",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "max_num_batched_tokens",
    "block_size",
    "max_num_seqs",
    "seed",
    "enable_expert_parallel",
    "enable_prefix_caching",
    "compilation_config",
)


def load_structured_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    data = yaml.safe_load(text) if yaml is not None else json.loads(_strip_json_comments(text))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain an object/mapping at top level")
    return data


def _strip_json_comments(text: str) -> str:
    """Remove YAML-style ``#`` comments from JSON-subset carrier files."""
    stripped_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut = len(line)
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if char == "#" and not in_string:
                cut = index
                break
        candidate = line[:cut].rstrip()
        if candidate.strip():
            stripped_lines.append(candidate)
    return "\n".join(stripped_lines)


def _normalize_engine(engine: str) -> str:
    value = str(engine or "").strip().lower()
    if value == "vllm-ascend":
        return "vllm_ascend"
    return value


def load_canonical_to_engine_mappings(config_root: str | Path = CONFIG_ROOT) -> dict[str, Any]:
    """Load Phase D canonical mapping document."""
    path = Path(config_root) / "mappings" / CANONICAL_MAPPING_FILE
    data = load_structured_file(path)
    canonical = data.get("canonical") or {}
    if not isinstance(canonical, dict):
        raise ValueError(f"{path} top-level canonical must be an object")
    return canonical


def get_engine_parameter_mapping(engine: str, *, config_root: str | Path = CONFIG_ROOT) -> dict[str, str]:
    """Return legacy-shaped canonical->engine field mapping for one engine.

    The returned dictionary replaces ``engine_parameter_mapping.json`` for runtime
    parameter translation while preserving the old call sites' expected shape.
    Missing engine mappings are represented as empty strings for known canonical
    keys, matching the old JSON behavior.
    """
    canonical = load_canonical_to_engine_mappings(config_root)
    engine_id = _normalize_engine(engine)
    mapping: dict[str, str] = {key: "" for key in COMMON_CANONICAL_KEYS}
    for canonical_key, engine_map in canonical.items():
        if not isinstance(engine_map, dict):
            logger.warning("Invalid canonical mapping for %s: expected object", canonical_key)
            continue
        item = engine_map.get(engine_id)
        if item is None and engine_id == "vllm_ascend":
            item = engine_map.get("vllm")
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        mapping[str(canonical_key)] = str(field) if field else ""
    return mapping
