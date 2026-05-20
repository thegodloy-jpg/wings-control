# -*- coding: utf-8 -*-
"""Phase D runtime loader for manifests, deviations, and env policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import os

try:  # pragma: no cover - deployment images may include PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from core.recipe_loader import version_matches

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class PhaseDRuntimeContext:
    engine: str
    version: str
    chip_id: str = ""
    allow_primary: bool = False


def load_structured_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _load_simple_yaml(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain an object/mapping at top level")
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value.replace("'", '"'))
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _strip_comment(line: str) -> str:
    in_string = False
    quote = ""
    for index, char in enumerate(line):
        if char in {'"', "'"}:
            if in_string and char == quote:
                in_string = False
                quote = ""
            elif not in_string:
                in_string = True
                quote = char
        elif char == "#" and not in_string:
            return line[:index]
    return line


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by Phase D carrier files."""
    result: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    current_list_key: str | None = None
    current_item: dict[str, Any] | None = None
    while index < len(lines):
        raw = _strip_comment(lines[index]).rstrip()
        index += 1
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            key = line[:-1].strip()
            result[key] = []
            current_list_key = key
            current_item = None
            continue
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = _parse_scalar(value)
            current_list_key = None
            current_item = None
            continue
        if current_list_key and line.startswith("- "):
            current_item = {}
            result[current_list_key].append(current_item)
            body = line[2:].strip()
            if body and ":" in body:
                key, value = body.split(":", 1)
                current_item[key.strip()] = _parse_scalar(value)
            continue
        if current_item is not None and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value in {">", "|"}:
                block: list[str] = []
                while index < len(lines):
                    block_raw = _strip_comment(lines[index]).rstrip()
                    block_indent = len(block_raw) - len(block_raw.lstrip(" "))
                    if block_raw.strip() and block_indent <= indent:
                        break
                    if block_raw.strip():
                        block.append(block_raw.strip())
                    index += 1
                current_item[key] = " ".join(block) if value == ">" else "\n".join(block)
            else:
                current_item[key] = _parse_scalar(value)
    return result


def _version_key(path: Path) -> tuple[int, ...]:
    parts = []
    for token in path.stem.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
    return tuple(parts or [0])


def _engine_dir(config_root: str | Path, engine: str) -> Path:
    return Path(config_root) / "manifests" / engine


def load_engine_manifest(ctx: PhaseDRuntimeContext, *, config_root: str | Path = CONFIG_ROOT) -> tuple[dict[str, Any], str | None]:
    """Load the best available manifest for an engine/version.

    Exact version wins.  If no exact file exists, the newest manifest for the
    engine is returned as a fallback so Phase D can still produce diagnostics.
    """
    manifest_dir = _engine_dir(config_root, ctx.engine)
    if not manifest_dir.is_dir():
        return {}, None
    if ctx.version:
        exact = manifest_dir / f"{ctx.version}.yaml"
        if exact.is_file():
            return load_structured_file(exact), str(exact)
    files = sorted(manifest_dir.glob("*.yaml"), key=_version_key, reverse=True)
    if not files:
        return {}, None
    return load_structured_file(files[0]), str(files[0])


def _applies_to_version(item: dict[str, Any], version: str) -> bool:
    return version_matches(version, item.get("applies_to_versions", "*"))


def _applies_to_hardware(item: dict[str, Any], chip_id: str) -> bool:
    hardware = item.get("applies_to_hardware")
    if not hardware:
        return True
    if not isinstance(hardware, list):
        return False
    return "*" in hardware or chip_id in hardware


def load_deviations(ctx: PhaseDRuntimeContext, *, config_root: str | Path = CONFIG_ROOT) -> tuple[list[dict[str, Any]], str | None]:
    path = Path(config_root) / "deviations" / f"{ctx.engine}.yaml"
    if not path.is_file():
        return [], None
    data = load_structured_file(path)
    deviations = data.get("deviations") or []
    if not isinstance(deviations, list):
        raise ValueError(f"{path} deviations must be a list")
    applicable = [
        item for item in deviations
        if isinstance(item, dict)
        and _applies_to_version(item, ctx.version)
        and _applies_to_hardware(item, ctx.chip_id)
    ]
    return applicable, str(path)


def load_env_policies(ctx: PhaseDRuntimeContext, *, config_root: str | Path = CONFIG_ROOT) -> tuple[list[dict[str, Any]], str | None]:
    path = Path(config_root) / "env_policies" / f"{ctx.engine}.yaml"
    if not path.is_file():
        return [], None
    data = load_structured_file(path)
    policies = data.get("env_policies") or []
    if not isinstance(policies, list):
        raise ValueError(f"{path} env_policies must be a list")
    applicable = [
        item for item in policies
        if isinstance(item, dict)
        and _applies_to_version(item, ctx.version)
        and _applies_to_hardware(item, ctx.chip_id)
    ]
    return applicable, str(path)


def build_phase_d_resolution(
    ctx: PhaseDRuntimeContext,
    current_params: dict[str, Any],
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> dict[str, Any]:
    """Resolve Phase D layer proposals without mutating current params."""
    manifest, manifest_ref = load_engine_manifest(ctx, config_root=config_root)
    deviations, deviations_ref = load_deviations(ctx, config_root=config_root)
    env_policies, env_policies_ref = load_env_policies(ctx, config_root=config_root)

    proposed_params: dict[str, Any] = {}
    for item in deviations:
        if item.get("target") != "cli":
            continue
        field = str(item.get("field") or "")
        if field:
            proposed_params[field] = item.get("wings_value")

    proposed_env: dict[str, Any] = {}
    for item in env_policies:
        name = str(item.get("name") or "")
        if not name:
            continue
        mode = item.get("mode")
        if mode == "idempotent" and name not in os.environ and "default_value" in item:
            proposed_env[name] = item.get("default_value")
        elif mode == "force_override" and "default_value" in item:
            proposed_env[name] = item.get("default_value")

    diff = {
        key: {
            "current_value": current_params.get(key),
            "phase_d_value": value,
            "would_change": current_params.get(key) != value,
        }
        for key, value in proposed_params.items()
    }
    return {
        "status": "ok",
        "context": {"engine": ctx.engine, "version": ctx.version, "chip_id": ctx.chip_id},
        "manifest": {"path": manifest_ref, "found": bool(manifest)},
        "deviations": {"path": deviations_ref, "items": deviations},
        "env_policies": {"path": env_policies_ref, "items": env_policies},
        "proposed_params": proposed_params,
        "proposed_env": proposed_env,
        "diff": diff,
        "summary": {
            "manifest_found": bool(manifest),
            "deviation_count": len(deviations),
            "env_policy_count": len(env_policies),
            "proposed_param_count": len(proposed_params),
            "proposed_env_count": len(proposed_env),
            "would_change_count": sum(1 for item in diff.values() if item["would_change"]),
        },
    }
