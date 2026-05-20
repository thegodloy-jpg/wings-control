# -*- coding: utf-8 -*-
"""Phase D runtime loaders for non-model configuration carriers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json
import logging

try:  # pragma: no cover - deployment images may include PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_DISTRIBUTED_DEFAULTS = {
    "master": {"host": "127.0.0.1", "port": 16000, "heartbeat_timeout": 90, "max_retries": 3},
    "workers": {"port": 16001},
    "scheduler": {"policy": "least_load", "fallback_policy": "round_robin"},
    "vllm_distributed": {"ray_head_port": 28020, "nixl_port": 5759, "rpc_port": 13355},
    "sglang_distributed": {"dist_port": 28030},
    "mindie_distributed": {"master_port": 27070},
}


def load_structured_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    data = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain an object/mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_distributed_runtime_config(*, config_root: str | Path = CONFIG_ROOT) -> dict[str, Any]:
    """Load Phase D distributed runtime defaults.

    New carrier: ``config/distributed/defaults.yaml``.  If it is unavailable,
    return built-in defaults instead of reading old ``config/defaults`` JSON.
    """
    path = Path(config_root) / "distributed" / "defaults.yaml"
    if not path.is_file():
        logger.warning("[phase-d] distributed defaults not found: %s; using built-in defaults", path)
        return deepcopy(_DISTRIBUTED_DEFAULTS)
    data = load_structured_file(path)
    return _deep_merge(_DISTRIBUTED_DEFAULTS, data)


def mindie_service_template_path() -> str:
    """Return container path for the MindIE service template."""
    return "/opt/wings-control/config/templates/mindie_service_config.json"
