# -*- coding: utf-8 -*-
"""Phase D runtime loader for engine-level default parameters."""

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
ENGINE_DEFAULTS_DIR = "engine_defaults"
_ENGINE_DEFAULTS_BUILTIN: dict[str, dict[str, Any]] = {
    "vllm": {
        "max_model_len": 4096,
        "trust_remote_code": True,
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "quantization": None,
        "quantization_param_path": None,
        "gpu_memory_utilization": 0.9,
        "enable_chunked_prefill": None,
        "block_size": 16,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
        "seed": 0,
        "enable_expert_parallel": None,
        "enable_prefix_caching": True,
    },
    "sglang": {
        "context_length": 4096,
        "trust_remote_code": True,
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "quantization": None,
        "quantization_param_path": None,
        "mem_fraction_static": 0.8,
        "enable_chunked_prefill": None,
        "max_running_requests": 256,
        "chunked_prefill_size": 4096,
        "random_seed": 42,
        "disable_chunked_prefix_cache": False,
    },
    "mindie": {
        "maxInputTokenLen": 2048,
        "maxIterTimes": 2048,
        "trustRemoteCode": True,
        "cacheBlockSize": 16,
        "maxBatchSize": 256,
        "maxPrefillBatchSize": 50,
        "maxPrefillTokens": 8192,
        "prefillTimeMsPerReq": 150,
        "prefillPolicyType": 0,
        "decodeTimeMsPerReq": 50,
        "decodePolicyType": 0,
        "maxPreemptCount": 0,
        "supportSelectBatch": False,
        "maxQueueDelayMicroseconds": 5000,
        "bufferResponseEnabled": False,
        "decodeExpectedTime": 50,
        "prefillExpectedTime": 1500,
        "enable_expert_parallel": None,
        "enable_prefix_caching": True,
        "httpsEnabled": False,
        "inferMode": "standard",
        "openAiSupport": "vllm",
        "tokenTimeout": 3600,
        "e2eTimeout": 65535,
        "allowAllZeroIpListening": False,
        "cpuMemSize": 5,
        "npuMemSize": -1,
    },
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


def normalize_engine_default_id(engine: str) -> str:
    value = str(engine or "vllm").strip().lower()
    if value in {"vllm_ascend", "vllm-ascend"}:
        return "vllm"
    return value


def engine_default_ref(engine: str, *, config_root: str | Path = CONFIG_ROOT) -> str:
    engine_id = normalize_engine_default_id(engine)
    return str(Path(config_root) / ENGINE_DEFAULTS_DIR / f"{engine_id}.yaml")


def load_engine_defaults(engine: str, *, config_root: str | Path = CONFIG_ROOT) -> dict[str, Any]:
    """Load Phase D engine-level defaults, with built-in safety fallback."""
    engine_id = normalize_engine_default_id(engine)
    path = Path(config_root) / ENGINE_DEFAULTS_DIR / f"{engine_id}.yaml"
    if path.is_file():
        return load_structured_file(path)
    defaults = _ENGINE_DEFAULTS_BUILTIN.get(engine_id, {})
    if defaults:
        logger.warning("[phase-d] engine defaults not found: %s; using built-in %s defaults", path, engine_id)
    return deepcopy(defaults)
