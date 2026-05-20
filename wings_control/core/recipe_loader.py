# -*- coding: utf-8 -*-
"""Phase A recipe shadow loader for v3.6.1.

This module intentionally does not mutate launcher parameters.  It scans the
new recipe directories, selects best matching architecture/model recipes, and
returns a shadow diff that can be written next to ``start_command.sh`` for audit
and migration work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import logging
import os
import re

try:  # pragma: no cover - exercised in deployment images with PyYAML installed
    import yaml  # type: ignore
except Exception:  # pragma: no cover - current local test env may not have PyYAML
    yaml = None

from core.hardware_detect import KNOWN_CHIP_IDS
from utils.file_utils import WriteOptions, safe_write_file

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
RECIPE_SHADOW_DIFF_FILENAME = "recipe_shadow_diff.json"


@dataclass(frozen=True)
class RecipeRuntimeContext:
    """Runtime facts used by the Phase A shadow matcher."""

    engine: str
    version: str
    chip_id: str
    model_name: str
    model_path: str
    architecture: str
    allow_experimental: bool = False


@dataclass
class RecipeCandidate:
    """A recipe that matched the runtime model or architecture."""

    path: str
    kind: str
    data: dict[str, Any]
    experimental: bool
    matched_by: str
    defaults: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def specificity(self) -> int:
        return 2 if self.kind == "model" else 1


@dataclass
class RecipeSelection:
    """Selected recipe plus diagnostics."""

    selected: RecipeCandidate | None
    candidates: list[RecipeCandidate]
    conflicts: list[RecipeCandidate] = field(default_factory=list)


def load_structured_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON document as a dictionary.

    PyYAML is used when present.  Without PyYAML, JSON syntax is accepted as a
    strict subset so unit tests and bootstrap environments remain usable.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text or "{}")
    if not isinstance(loaded, dict):
        raise ValueError(f"{p} must contain a mapping/object at top level")
    return loaded


def read_model_architecture(model_path: str) -> str:
    """Read ``architectures[0]`` from a HuggingFace-style config.json."""
    if not model_path:
        return ""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return ""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[recipe-shadow] failed to read %s: %s", config_path, exc)
        return ""
    architectures = data.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0] or "")
    return ""


def _casefold(value: Any) -> str:
    return str(value or "").strip().casefold()


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(raw or ""))
    return tuple(int(p) for p in parts[:4])


def _compare_versions(left: str, right: str) -> int:
    a = _version_tuple(left)
    b = _version_tuple(right)
    max_len = max(len(a), len(b), 1)
    a = a + (0,) * (max_len - len(a))
    b = b + (0,) * (max_len - len(b))
    return (a > b) - (a < b)


def version_matches(version: str, specifier: Any) -> bool:
    """Evaluate the small PEP 440 subset used by recipe matching."""
    spec = str(specifier or "*").strip()
    if spec in {"", "*"}:
        return True
    if not version:
        return False
    for clause in [p.strip() for p in spec.split(",") if p.strip()]:
        if clause.startswith(">="):
            if _compare_versions(version, clause[2:].strip()) < 0:
                return False
        elif clause.startswith("<="):
            if _compare_versions(version, clause[2:].strip()) > 0:
                return False
        elif clause.startswith(">"):
            if _compare_versions(version, clause[1:].strip()) <= 0:
                return False
        elif clause.startswith("<"):
            if _compare_versions(version, clause[1:].strip()) >= 0:
                return False
        elif clause.startswith("=="):
            if _compare_versions(version, clause[2:].strip()) != 0:
                return False
        elif clause.startswith("!="):
            if _compare_versions(version, clause[2:].strip()) == 0:
                return False
    return True


def _iter_recipe_files(root: Path, kind: str, allow_experimental: bool) -> list[Path]:
    base = root / "recipes" / ("architectures" if kind == "architecture" else "models")
    files = sorted(p for p in base.glob("*.yaml") if p.is_file())
    if allow_experimental:
        files.extend(sorted(p for p in (base / "_experimental").glob("*.yaml") if p.is_file()))
    return files


def _hardware_overlay_defaults(data: dict[str, Any], chip_id: str) -> dict[str, Any]:
    merged: dict[str, Any] = dict(data.get("defaults") or {})
    overlays = data.get("hardware_overlays") or []
    if not isinstance(overlays, list):
        return merged
    for overlay in overlays:
        if not isinstance(overlay, dict):
            continue
        hardware = overlay.get("applies_to_hardware") or []
        if "*" in hardware or chip_id in hardware:
            merged.update(overlay.get("defaults") or {})
    return merged


def _is_experimental_path(path: Path) -> bool:
    return "_experimental" in path.parts


def _applies_to_version(data: dict[str, Any], engine: str, version: str) -> bool:
    spec = data.get("applies_to_versions")
    if isinstance(spec, dict):
        clause = spec.get(engine)
    else:
        clause = spec
    return version_matches(version, clause)


def _matches_architecture(data: dict[str, Any], architecture: str) -> bool:
    arch = _casefold(architecture)
    if not arch:
        return False
    values = [data.get("architecture"), *(data.get("also_matches") or [])]
    return arch in {_casefold(v) for v in values if v}


def _matches_model(data: dict[str, Any], model_name: str, model_path: str) -> bool:
    model_name_cf = _casefold(model_name)
    path_cf = _casefold(model_path)
    base_cf = _casefold(Path(model_path).name)
    model_id_cf = _casefold(data.get("model_id"))
    if model_id_cf and model_id_cf in {model_name_cf, base_cf}:
        return True
    matches = data.get("matches") or {}
    if not isinstance(matches, dict):
        return False
    for item in matches.get("model_names") or []:
        if _casefold(item) == model_name_cf:
            return True
    for item in matches.get("model_paths") or []:
        needle = _casefold(item)
        if needle and (needle in path_cf or needle == base_cf):
            return True
    return False


def compute_match_score(candidate: RecipeCandidate, ctx: RecipeRuntimeContext) -> float:
    """Compute the match score for one candidate.

    Signals:
      - ``version_match`` (30): candidate declares ``applies_to_versions`` and
        the runtime engine version satisfies it.
      - ``hardware_match`` (100): a ``hardware_overlays`` entry applies to the
        runtime chip_id.
      - ``specificity`` (model=200, architecture=100): model recipes win when
        present.
    """
    breakdown = {
        "version_match": 0.0,
        "hardware_match": 0.0,
        "specificity": 0.0,
    }
    if _applies_to_version(candidate.data, ctx.engine, ctx.version):
        breakdown["version_match"] = 30.0
    for overlay in candidate.data.get("hardware_overlays") or []:
        if not isinstance(overlay, dict):
            continue
        hardware = overlay.get("applies_to_hardware") or []
        if "*" in hardware or (ctx.chip_id and ctx.chip_id in hardware):
            breakdown["hardware_match"] = 100.0
            break
    breakdown["specificity"] = 200.0 if candidate.kind == "model" else 100.0
    candidate.score_breakdown = breakdown
    return sum(breakdown.values())


def _candidate_from_file(path: Path, kind: str, ctx: RecipeRuntimeContext) -> RecipeCandidate | None:
    try:
        data = load_structured_file(path)
    except Exception as exc:
        logger.warning("[recipe-shadow] failed to load %s: %s", path, exc)
        return None
    if kind == "architecture":
        if not _matches_architecture(data, ctx.architecture):
            return None
        matched_by = "architecture"
    else:
        if not _matches_model(data, ctx.model_name, ctx.model_path):
            return None
        matched_by = "model"
    candidate = RecipeCandidate(
        path=str(path),
        kind="architecture" if kind == "architecture" else "model",
        data=data,
        experimental=_is_experimental_path(path),
        matched_by=matched_by,
        defaults=_hardware_overlay_defaults(data, ctx.chip_id),
    )
    candidate.score = compute_match_score(candidate, ctx)
    return candidate


def find_recipe_candidates(ctx: RecipeRuntimeContext, config_root: str | Path = CONFIG_ROOT) -> tuple[list[RecipeCandidate], list[RecipeCandidate]]:
    """Find architecture and model recipes that match the runtime context."""
    root = Path(config_root)
    arch_candidates: list[RecipeCandidate] = []
    model_candidates: list[RecipeCandidate] = []
    for path in _iter_recipe_files(root, "architecture", ctx.allow_experimental):
        candidate = _candidate_from_file(path, "architecture", ctx)
        if candidate:
            arch_candidates.append(candidate)
    for path in _iter_recipe_files(root, "model", ctx.allow_experimental):
        candidate = _candidate_from_file(path, "model", ctx)
        if candidate:
            model_candidates.append(candidate)
    return arch_candidates, model_candidates


def select_best_recipe(candidates: list[RecipeCandidate]) -> RecipeSelection:
    """Select the highest-scoring recipe and report unresolved ties."""
    if not candidates:
        return RecipeSelection(selected=None, candidates=[])
    ordered = sorted(
        candidates,
        key=lambda c: (c.score, c.specificity, c.path),
        reverse=True,
    )
    selected = ordered[0]
    conflicts = [
        c for c in ordered[1:]
        if c.score == selected.score and c.specificity == selected.specificity
    ]
    return RecipeSelection(selected=selected, candidates=ordered, conflicts=conflicts)


def _candidate_summary(candidate: RecipeCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "path": candidate.path,
        "kind": candidate.kind,
        "experimental": candidate.experimental,
        "matched_by": candidate.matched_by,
        "score": candidate.score,
        "score_breakdown": candidate.score_breakdown,
        "defaults": candidate.defaults,
    }


def _merged_recipe_defaults(arch: RecipeCandidate | None, model: RecipeCandidate | None) -> dict[str, Any]:
    proposed: dict[str, Any] = {}
    if arch:
        proposed.update(arch.defaults)
    if model:
        proposed.update(model.defaults)
    return proposed


def resolve_recipe_defaults(
    ctx: RecipeRuntimeContext,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve recipe defaults and selection diagnostics for a runtime context.

    Phase B reuses this helper before the legacy ``model_deploy_config`` lookup.
    It only reads recipe files and returns proposed defaults plus metadata; callers
    decide whether to apply or only report the result.
    """
    arch_candidates, model_candidates = find_recipe_candidates(ctx, config_root=config_root)
    arch_selection = select_best_recipe(arch_candidates)
    model_selection = select_best_recipe(model_candidates)
    selected_candidates = [c for c in (arch_selection.selected, model_selection.selected) if c]
    proposed = _merged_recipe_defaults(arch_selection.selected, model_selection.selected)
    status = "conflict" if arch_selection.conflicts or model_selection.conflicts else "ok"
    return proposed, {
        "status": status,
        "summary": {
            "selected_count": len(selected_candidates),
            "proposed_param_count": len(proposed),
            "experimental_selected": any(c.experimental for c in selected_candidates),
            "conflict_count": len(arch_selection.conflicts) + len(model_selection.conflicts),
        },
        "selected": {
            "architecture": _candidate_summary(arch_selection.selected),
            "model": _candidate_summary(model_selection.selected),
        },
        "conflicts": {
            "architecture": [_candidate_summary(c) for c in arch_selection.conflicts],
            "model": [_candidate_summary(c) for c in model_selection.conflicts],
        },
        "diagnostics": {
            "architecture_candidates": [_candidate_summary(c) for c in arch_selection.candidates],
            "model_candidates": [_candidate_summary(c) for c in model_selection.candidates],
            "experimental_scan_enabled": ctx.allow_experimental,
        },
    }


def build_recipe_shadow_diff(
    launch_args: Any,
    hardware: dict[str, Any],
    merged_params: dict[str, Any],
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> dict[str, Any]:
    """Build a Phase A recipe shadow diff without changing ``merged_params``."""
    mode = os.getenv("WINGS_RECIPE_MODE", "shadow").strip().lower() or "shadow"
    if mode == "off":
        return {"mode": "off", "effective_mode": "off", "status": "disabled"}

    model_path = str(getattr(launch_args, "model_path", "") or merged_params.get("model_path") or "")
    architecture = read_model_architecture(model_path)
    ctx = RecipeRuntimeContext(
        engine=str(merged_params.get("engine") or getattr(launch_args, "engine", "")),
        version=str(merged_params.get("engine_version") or merged_params.get("version") or ""),
        chip_id=str(hardware.get("chip_id") or ""),
        model_name=str(getattr(launch_args, "model_name", "") or merged_params.get("model_name") or ""),
        model_path=model_path,
        architecture=architecture,
        allow_experimental=bool(getattr(launch_args, "allow_experimental", False)),
    )
    proposed, resolution = resolve_recipe_defaults(ctx, config_root=config_root)
    diff = {
        key: {
            "legacy_value": merged_params.get(key),
            "recipe_value": value,
            "would_change": merged_params.get(key) != value,
        }
        for key, value in proposed.items()
    }
    would_change_count = sum(1 for item in diff.values() if item["would_change"])
    summary = dict(resolution.get("summary") or {})
    summary["would_change_count"] = would_change_count
    return {
        "mode": mode,
        "effective_mode": "shadow",
        "status": resolution.get("status", "ok"),
        "phase": "A",
        "summary": summary,
        "context": {
            "engine": ctx.engine,
            "version": ctx.version,
            "chip_id": ctx.chip_id,
            "model_name": ctx.model_name,
            "model_path": ctx.model_path,
            "architecture": ctx.architecture,
            "allow_experimental": ctx.allow_experimental,
        },
        "selected": resolution.get("selected", {}),
        "conflicts": resolution.get("conflicts", {}),
        "diagnostics": {
            **(resolution.get("diagnostics") or {}),
            "recipe_mode_note": "Phase A never mutates final merged params; primary mode is reported but treated as shadow.",
        },
        "proposed_params": proposed,
        "diff": diff,
    }


def write_recipe_shadow_diff(
    launch_args: Any,
    hardware: dict[str, Any],
    merged_params: dict[str, Any],
    *,
    config_root: str | Path = CONFIG_ROOT,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write Phase A shadow diff to the shared volume and return it."""
    shadow = build_recipe_shadow_diff(launch_args, hardware, merged_params, config_root=config_root)
    if shadow.get("effective_mode") == "off":
        logger.info("[recipe-shadow] disabled by WINGS_RECIPE_MODE=off")
        return shadow
    out_dir = Path(output_dir or os.getenv("SHARED_VOLUME_PATH", "/shared-volume"))
    out_path = out_dir / RECIPE_SHADOW_DIFF_FILENAME
    ok = safe_write_file(
        str(out_path),
        shadow,
        is_json=True,
        options=WriteOptions(is_json=True, atomic=True),
    )
    if ok:
        logger.info("[recipe-shadow] wrote %s", out_path)
    else:
        logger.warning("[recipe-shadow] failed to write %s", out_path)
    summary = shadow.get("summary") or {}
    selected = shadow.get("selected") or {}
    logger.info(
        "[recipe-shadow] status=%s selected=%s proposed=%s would_change=%s",
        shadow.get("status"),
        summary.get("selected_count", 0),
        summary.get("proposed_param_count", 0),
        summary.get("would_change_count", 0),
    )
    if summary.get("experimental_selected"):
        logger.warning(
            "[recipe-shadow] experimental recipe selected in shadow mode: arch=%s model=%s",
            (selected.get("architecture") or {}).get("path"),
            (selected.get("model") or {}).get("path"),
        )
    if shadow.get("status") == "conflict":
        logger.warning("[recipe-shadow] recipe selection conflicts detected; see %s", out_path)
    return shadow


def validate_recipe_document(path: str | Path, data: dict[str, Any], known_chips: set[str] | None = None) -> list[tuple[str, str]]:
    """Return lint diagnostics as ``(level, message)`` tuples."""
    known_chips = known_chips or set(KNOWN_CHIP_IDS)
    p = Path(path)
    diagnostics: list[tuple[str, str]] = []
    is_architecture_recipe = "architectures" in p.parts
    if is_architecture_recipe and not data.get("architecture"):
        diagnostics.append(("error", "architecture recipe missing 'architecture' field"))
    for overlay in data.get("hardware_overlays") or []:
        if not isinstance(overlay, dict):
            continue
        for chip in overlay.get("applies_to_hardware") or []:
            if chip != "*" and chip not in known_chips:
                diagnostics.append(("error", f"hardware_overlays references unknown hardware: {chip}"))
    return diagnostics
