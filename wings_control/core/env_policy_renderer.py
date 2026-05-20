# -*- coding: utf-8 -*-
"""Render env_policies entries as ``export`` shell commands using runtime context.

Phase D Step 2B carrier: extends env_policies/<engine>.yaml with two optional
fields so that adapter code stops emitting hand-written ``export`` strings:

- ``value_template``: literal text with ``{{key}}`` placeholders. Each placeholder
  is substituted with ``context[key]`` (or with ``env(NAME, default)`` reading
  ``os.environ``). Substituted values are shell-quoted; literal text — including
  bash-native expressions like ``${LD_LIBRARY_PATH:-}`` — is preserved verbatim.

- ``applies_when``: AND-predicate dict. Each (key, expected) pair must match
  ``context[key]``. Scalar expected uses equality (bool expected uses truthiness
  on either side). List expected uses membership.

Entries without either field keep their existing Phase D behavior; this module
is additive.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from core.phase_d_loader import CONFIG_ROOT, load_structured_file

_BRACE_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_DOLLAR_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_FN_RE = re.compile(r"^env\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*(.*?)\s*)?\)$")


def _resolve_placeholder(token: str, context: dict[str, Any]) -> str:
    match = _ENV_FN_RE.match(token)
    if match:
        name = match.group(1)
        default = match.group(2) or ""
        if default.startswith('"') and default.endswith('"'):
            default = default[1:-1]
        elif default.startswith("'") and default.endswith("'"):
            default = default[1:-1]
        return os.getenv(name, default)
    if token in context and context[token] is not None:
        return str(context[token])
    return ""


def _expand_template(template: str, context: dict[str, Any]) -> str:
    def brace_repl(match: re.Match[str]) -> str:
        raw = _resolve_placeholder(match.group(1).strip(), context)
        return shlex.quote(raw) if raw else "''"

    def dollar_repl(match: re.Match[str]) -> str:
        raw = _resolve_placeholder(match.group(1).strip(), context)
        return shlex.quote(raw) if raw else "''"

    expanded = _BRACE_PLACEHOLDER_RE.sub(brace_repl, template)
    return _DOLLAR_PLACEHOLDER_RE.sub(dollar_repl, expanded)


def _predicate_matches(applies_when: dict[str, Any], context: dict[str, Any]) -> bool:
    if not applies_when:
        return True
    for key, expected in applies_when.items():
        actual = context.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif isinstance(expected, bool):
            if bool(actual) != expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def render_env_exports(
    env_policies: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    include_default_values: bool = False,
) -> list[str]:
    """Return ``export NAME=value`` commands for entries matching context.

    Iteration order matches the order in the yaml carrier, so adapters get a
    deterministic export sequence.
    """
    commands: list[str] = []
    for item in env_policies:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        if item.get("mode") == "inherit":
            continue
        if not _predicate_matches(item.get("applies_when") or {}, context):
            continue
        if "value_template" in item:
            value = _expand_template(str(item["value_template"]), context)
            commands.append(f"export {name}={value}")
        elif include_default_values and "default_value" in item:
            value = str(item["default_value"])
            commands.append(f"export {name}={shlex.quote(value)}")
    return commands


def load_engine_env_policies(
    engine: str,
    *,
    config_root: Path | str = CONFIG_ROOT,
) -> list[dict[str, Any]]:
    """Load env_policies/<engine>.yaml, returning the policy list (or empty)."""
    path = Path(config_root) / "env_policies" / f"{engine}.yaml"
    if not path.is_file():
        return []
    data = load_structured_file(path)
    items = data.get("env_policies") or []
    return [item for item in items if isinstance(item, dict)]
