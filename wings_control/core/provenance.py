# -*- coding: utf-8 -*-
"""Field-level provenance helpers for resolved engine parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import logging
import os

from utils.file_utils import WriteOptions, safe_write_file

logger = logging.getLogger(__name__)

RESOLVED_PARAMS_FILENAME = "resolved_params.json"


class LoadResult(dict):
    """Backward-compatible structured result for config loading.

    The object behaves like the legacy merged parameter dict while also exposing
    Phase C structured attributes.  Existing call sites can keep using
    ``result["engine_config"]``; new call sites can use ``result.merged`` and
    ``result.resolution_trace``.
    """

    def __init__(
        self,
        merged: dict[str, Any],
        *,
        resolution_trace: dict[str, dict[str, Any]] | None = None,
        chip_info: dict[str, Any] | None = None,
        matched_recipes: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(merged)
        self.merged = self
        self.resolution_trace = resolution_trace or {}
        self.chip_info = chip_info or {}
        self.matched_recipes = matched_recipes or []

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy merged dict view."""
        return self.merged


@dataclass(frozen=True)
class ParamDecision:
    """Four-key resolved_params.json decision record."""

    value: Any
    target: str
    source_layer: str
    source_ref: str | None


def _target_for_key(key: str) -> str:
    if key.startswith("env.") or key.isupper():
        return key if key.startswith("env.") else f"env.{key}"
    if key.startswith("additional_config.") or key.startswith("generated_file.") or key.startswith("cli."):
        return key
    return f"cli.{key}"


def _recipe_decision_for_key(key: str, value: Any, recipe_resolution: dict[str, Any] | None) -> ParamDecision | None:
    if not recipe_resolution:
        return None
    selected = recipe_resolution.get("selected") or {}
    model = selected.get("model") if isinstance(selected, dict) else None
    arch = selected.get("architecture") if isinstance(selected, dict) else None
    for source_layer, candidate in (("model_recipe", model), ("arch_recipe", arch)):
        if not isinstance(candidate, dict):
            continue
        defaults = candidate.get("defaults") or {}
        if key not in defaults:
            continue
        return ParamDecision(
            value=value,
            target=_target_for_key(key),
            source_layer=source_layer,
            source_ref=f"{candidate.get('path')}#{key}" if candidate.get("path") else None,
        )
    return None


def _phase_d_deviation_decision_for_key(
    key: str,
    value: Any,
    phase_d_resolution: dict[str, Any] | None,
) -> ParamDecision | None:
    if not phase_d_resolution:
        return None
    proposed = phase_d_resolution.get("proposed_params") or {}
    if key not in proposed or proposed.get(key) != value:
        return None
    deviations = ((phase_d_resolution.get("deviations") or {}).get("items") or [])
    source_path = (phase_d_resolution.get("deviations") or {}).get("path")
    for item in deviations:
        if not isinstance(item, dict):
            continue
        if item.get("target") == "cli" and item.get("field") == key:
            return ParamDecision(
                value=value,
                target=_target_for_key(key),
                source_layer="deviation",
                source_ref=f"{source_path}#{key}" if source_path else key,
            )
    return None


def _phase_d_env_policy_decisions(phase_d_resolution: dict[str, Any] | None) -> dict[str, ParamDecision]:
    if not phase_d_resolution:
        return {}
    proposed_env = phase_d_resolution.get("proposed_env") or {}
    source_path = (phase_d_resolution.get("env_policies") or {}).get("path")
    decisions: dict[str, ParamDecision] = {}
    for name, value in proposed_env.items():
        key = f"env.{name}"
        decisions[key] = ParamDecision(
            value=value,
            target=key,
            source_layer="env_policy",
            source_ref=f"{source_path}#{name}" if source_path else name,
        )
    return decisions


def build_resolution_trace(
    engine_config: dict[str, Any],
    *,
    explicit_keys: set[str] | None = None,
    user_config_keys: set[str] | None = None,
    user_config_ref: str | None = None,
    recipe_resolution: dict[str, Any] | None = None,
    phase_d_resolution: dict[str, Any] | None = None,
    legacy_ref: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a seven-key decision map for final engine_config fields."""
    explicit_keys = explicit_keys or set()
    user_config_keys = user_config_keys or set()
    trace: dict[str, dict[str, Any]] = {}

    for key, value in engine_config.items():
        if str(key).startswith("_"):
            continue
        namespaced_key = _target_for_key(str(key))
        if key in explicit_keys:
            decision = ParamDecision(
                value=value,
                target=namespaced_key,
                source_layer="user_cli",
                source_ref=f"--{str(key).replace('_', '-')}",
            )
        elif key in user_config_keys:
            decision = ParamDecision(
                value=value,
                target=namespaced_key,
                source_layer="config_file",
                source_ref=f"{user_config_ref}#{key}" if user_config_ref else str(key),
            )
        else:
            recipe_decision = _recipe_decision_for_key(str(key), value, recipe_resolution)
            phase_d_decision = _phase_d_deviation_decision_for_key(str(key), value, phase_d_resolution)
            decision = recipe_decision or phase_d_decision or ParamDecision(
                value=value,
                target=namespaced_key,
                source_layer="legacy_model_deploy_config",
                source_ref=f"{legacy_ref}#{key}" if legacy_ref else str(key),
            )
        trace[namespaced_key] = asdict(decision)
    for target, decision in _phase_d_env_policy_decisions(phase_d_resolution).items():
        trace.setdefault(target, asdict(decision))
    return trace


def write_resolved_params(
    trace: dict[str, dict[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> Path | None:
    """Write resolved_params.json when explicitly requested."""
    if not trace:
        logger.info("[resolved-params] no resolution trace to write")
        return None
    path = Path(output_path) if output_path else Path(os.getenv("SHARED_VOLUME_PATH", "/shared-volume")) / RESOLVED_PARAMS_FILENAME
    ok = safe_write_file(
        str(path),
        trace,
        is_json=True,
        options=WriteOptions(is_json=True, atomic=True),
    )
    if ok:
        logger.info("[resolved-params] wrote %s", path)
        return path
    logger.warning("[resolved-params] failed to write %s", path)
    return None
