#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for v3.6.1 Phase D config tooling.

The repository currently keeps bootstrap YAML files in a JSON-compatible subset
so the tools can run without PyYAML.  If PyYAML is available, regular YAML is
accepted as well.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - deployment images may include PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover - local bootstrap environment may not
    yaml = None

SUPPORTED_ENGINES = {
    "vllm", "vllm_ascend", "vllm_distributed", "vllm_ascend_distributed",
    "sglang", "sglang_distributed",
    "mindie", "mindie_distributed",
}
SUPPORTED_TARGETS = {"cli", "env", "additional_config", "generated_file"}
SUPPORTED_SOURCE_TYPES = {
    "official_doc",
    "engine_source",
    "vendor_doc",
    "vendor_cookbook",
    "vendor_se_run",
    "wings_validation_run",
    "wings_decision",
}
LEGACY_SOURCE_TYPES = {"wings_internal", "vendor_validated"}
SUPPORTED_ENV_MODES = {"idempotent", "force_override", "inherit"}
KNOWN_CHIP_IDS = {
    "h20-96",
    "h20-141",
    "l20",
    "rtx-pro-5000",
    "910b-32",
    "910b-64",
    "910c",
}
HARDWARE_ALIASES = {
    "910b-32g": "910b-32",
    "910b_32g": "910b-32",
    "910b-64g": "910b-64",
    "910b_64g": "910b-64",
    "rtx_pro_5000": "rtx-pro-5000",
}

Diagnostic = tuple[str, str]


def load_structured_file(path: str | Path) -> dict[str, Any]:
    """Load JSON/YAML as a mapping without requiring PyYAML."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        try:
            data = json.loads(_strip_json_comments(text))
        except json.JSONDecodeError:
            data = _load_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain an object/mapping at top level")
    return data


def _strip_comment(line: str) -> str:
    in_string = False
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
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


def _strip_json_comments(text: str) -> str:
    return "\n".join(
        candidate
        for candidate in (_strip_comment(line).rstrip() for line in text.splitlines())
        if candidate.strip()
    )


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


def _load_simple_yaml(text: str) -> dict[str, Any]:
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


def write_structured_file(path: str | Path, data: dict[str, Any]) -> None:
    """Write deterministic JSON-subset YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_config_files(directory: str | Path) -> list[Path]:
    """Return sorted YAML/JSON files directly under a directory."""
    root = Path(directory)
    if not root.is_dir():
        return []
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml", "*.json"):
        files.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(files))


def _has_value(data: dict[str, Any], key: str) -> bool:
    return key in data and data[key] not in (None, "", [])


def require_fields(item: dict[str, Any], fields: list[str], label: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for field in fields:
        if not _has_value(item, field):
            diagnostics.append(("error", f"{label} missing {field}"))
    return diagnostics


def validate_source_type(item: dict[str, Any], label: str) -> list[Diagnostic]:
    source_type = str(item.get("source_type") or "")
    if not source_type:
        return [("error", f"{label} missing source_type")]
    if source_type in LEGACY_SOURCE_TYPES:
        return [("warning", f"{label} uses legacy source_type: {source_type}")]
    if source_type not in SUPPORTED_SOURCE_TYPES:
        return [("error", f"{label} invalid source_type: {source_type!r}")]
    return []


def validate_hardware_list(item: dict[str, Any], field: str, label: str) -> list[Diagnostic]:
    values = item.get(field)
    if values in (None, ""):
        return []
    if not isinstance(values, list):
        return [("error", f"{label}.{field} must be a list")]
    diagnostics: list[Diagnostic] = []
    for value in values:
        chip = str(value)
        if chip == "*" or chip in KNOWN_CHIP_IDS:
            continue
        if chip in HARDWARE_ALIASES:
            diagnostics.append(("warning", f"{label}.{field} uses alias {chip}; use {HARDWARE_ALIASES[chip]}"))
        else:
            diagnostics.append(("error", f"{label}.{field} references unknown hardware: {chip}"))
    return diagnostics


def summarize_results(results: list[dict[str, Any]]) -> tuple[int, int]:
    errors = 0
    warnings = 0
    for result in results:
        for diagnostic in result.get("diagnostics", []):
            if diagnostic.get("level") == "error":
                errors += 1
            elif diagnostic.get("level") == "warning":
                warnings += 1
    return errors, warnings


def print_results(results: list[dict[str, Any]], *, output_json: bool) -> int:
    errors, warnings = summarize_results(results)
    if output_json:
        print(json.dumps({
            "files_checked": len(results),
            "errors": errors,
            "warnings": warnings,
            "results": results,
        }, ensure_ascii=False, indent=2))
    else:
        for result in results:
            for diagnostic in result.get("diagnostics", []):
                prefix = "ERROR" if diagnostic.get("level") == "error" else "WARN "
                print(f"{prefix}  {result['file']}: {diagnostic.get('message')}")
        print(f"\n{len(results)} file(s) checked — {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0
