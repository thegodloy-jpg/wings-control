#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline-friendly upstream sync helpers for official_sources.yaml.

Honors the official_sources.yaml contract:
  - runtime_remote_fetch: false → fetching is only done from this tool, never
    from the launcher hot path.
  - default_mode: report → this module never overwrites project yaml; it
    writes diff reports under build/official_start_command_refs/.

Layering
--------
  Fetcher  : URL → str           (urllib, with on-disk cache so reruns are free)
  Parser   : str  → structured   (one parser per source.id; html.parser stdlib)
  Differ   : structured + local  → list[dict] of additions/removals/value drift
  Reporter : list[dict]          → write JSON under build/official_start_command_refs/

All parsers are tolerant: anything they can't classify is dropped, never raised.
This keeps the tool useful when upstream HTML rearranges.
"""

from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from _config_tooling import load_structured_file


@dataclass
class FetchResult:
    source_id: str
    url: str
    body: str
    from_cache: bool
    error: str | None = None


@dataclass
class ParseResult:
    source_id: str
    kind: str
    items: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class DiffEntry:
    field: str
    status: str  # one of: upstream_only, local_only, value_drift
    upstream: Any = None
    local: Any = None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_url(
    source_id: str,
    url: str,
    *,
    cache_dir: Path,
    use_cache: bool,
    fixture_dir: Path | None = None,
    timeout: float = 30.0,
    opener: Callable[[str], str] | None = None,
) -> FetchResult:
    """Fetch a URL with on-disk cache. Honors fixture_dir for offline runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{source_id}.html"

    if fixture_dir is not None:
        candidate = fixture_dir / f"{source_id}.html"
        if candidate.is_file():
            return FetchResult(source_id, url, candidate.read_text(encoding="utf-8"), from_cache=True)

    if use_cache and cache_file.is_file():
        return FetchResult(source_id, url, cache_file.read_text(encoding="utf-8"), from_cache=True)

    try:
        if opener is not None:
            body = opener(url)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "wings-control-sync/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        return FetchResult(source_id, url, "", from_cache=False, error=str(exc))

    cache_file.write_text(body, encoding="utf-8")
    return FetchResult(source_id, url, body, from_cache=False)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip tags, keep text. Used for table cells and code blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    @property
    def text(self) -> str:
        return "".join(self._chunks)


def _strip_tags(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text


class _CodeBlockExtractor(HTMLParser):
    """Extract text from ``<pre>`` blocks while preserving code whitespace."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[str] = []
        self._pre_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            self._pre_depth += 1
            if self._pre_depth == 1:
                self._chunks = []
        elif tag == "br" and self._pre_depth:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if self._pre_depth == 0:
                self.blocks.append("".join(self._chunks))
                self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._pre_depth:
            self._chunks.append(data)


def _extract_code_blocks(html: str) -> list[str]:
    parser = _CodeBlockExtractor()
    parser.feed(html)
    return parser.blocks


# Light table extractor: returns list of rows, each row a list of cell text.
class _TableExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._stack: list[str] = []
        self._cell_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs):  # noqa: D401
        if tag == "table":
            self.tables.append([])
            self._stack.append("table")
        elif tag == "tr" and self._stack and self._stack[-1] in {"table", "thead", "tbody"}:
            self.tables[-1].append([])
            self._stack.append("tr")
        elif tag in {"thead", "tbody"} and self._stack and self._stack[-1] == "table":
            self._stack.append(tag)
        elif tag in {"td", "th"} and self._stack and self._stack[-1] == "tr":
            self._cell_buf = []
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        if tag in {"td", "th"} and self._stack[-1] in {"td", "th"}:
            self.tables[-1][-1].append("".join(self._cell_buf).strip())
            self._cell_buf = []
            self._stack.pop()
        elif tag == "tr" and self._stack[-1] == "tr":
            self._stack.pop()
        elif tag in {"thead", "tbody"} and self._stack[-1] == tag:
            self._stack.pop()
        elif tag == "table" and self._stack[-1] == "table":
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._stack and self._stack[-1] in {"td", "th"}:
            self._cell_buf.append(data)


def _extract_tables(html: str) -> list[list[list[str]]]:
    parser = _TableExtractor()
    parser.feed(html)
    return parser.tables


# ---------------------------------------------------------------------------
# Parsers — one per source.id in official_sources.yaml
# ---------------------------------------------------------------------------

# vllm CLI reference (docs.vllm.ai/en/latest/cli/serve/, mkdocs-material):
#   <h4 id=-flag-name><code>--flag-name</code>...</h4>
#   <dl> <dd>description...</dd> <dd>Default: <code>VALUE</code></dd> </dl>
# Some options have no Default: line (required args or dynamic defaults).
# Group headings like <h3 id="cacheconfig">CacheConfig</h3> split dataclasses.
_VLLM_FLAG_HEADER = re.compile(
    r"<h4\b(?=[^>]*\bid=(?:\"?-[a-z0-9-]+\"?|'?-[a-z0-9-]+'?))[^>]*>"
    r"(?P<header>.*?)</h4>",
    re.IGNORECASE | re.DOTALL,
)
_VLLM_HEADER_FLAG = re.compile(r"<code[^>]*>\s*(?P<flag>--[a-z0-9][a-z0-9-]*)\s*</code>", re.IGNORECASE)
_VLLM_BODY_DEFAULT = re.compile(
    r"<dd>\s*Default:\s*<code[^>]*>(?P<default>[^<]*)</code>\s*</dd>",
    re.IGNORECASE,
)
_VLLM_BODY_DEPRECATED = re.compile(r"deprecat", re.IGNORECASE)


def parse_vllm_cli_serve(html: str) -> ParseResult:
    """Extract CLI parameters and defaults from docs.vllm.ai/cli/serve/."""
    result = ParseResult(source_id="vllm-cli-serve-reference", kind="cli_params")
    headers = list(_VLLM_FLAG_HEADER.finditer(html))
    for idx, match in enumerate(headers):
        header_flags = [m.group("flag") for m in _VLLM_HEADER_FLAG.finditer(match.group("header"))]
        if not header_flags:
            continue
        flag_token = next((flag for flag in header_flags if not flag.startswith("--no-")), header_flags[0])
        flag_raw = flag_token[2:]  # strip leading "--"
        flag = flag_raw.replace("-", "_").lower()
        if flag in result.items:
            continue
        body_start = match.end()
        body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(html)
        body = html[body_start:body_end]
        default_match = _VLLM_BODY_DEFAULT.search(body)
        default_raw = default_match.group("default").strip() if default_match else None
        has_default = default_raw is not None
        result.items[flag] = {
            "upstream_default": _coerce_default(default_raw),
            "has_default": has_default,
            "deprecated": bool(_VLLM_BODY_DEPRECATED.search(_strip_tags(body)[:400])),
        }
    if not result.items:
        result.notes.append("no CLI options parsed; HTML structure may have changed")
    return result


# vllm-ascend env vars are currently documented as a highlighted Python dict:
#   "VLLM_ASCEND_FOO": lambda: bool(int(os.getenv("VLLM_ASCEND_FOO", "0"))),
# Older fixtures may still be tables with NAME | Default | Description.
_ENV_ENTRY = re.compile(
    r'^\s*"(?P<name>[A-Z][A-Z0-9_]*)"\s*:\s*lambda\s*:\s*(?P<body>.*?)'
    r'(?=^\s*"[A-Z][A-Z0-9_]*"\s*:\s*lambda\s*:|^\s*}\s*,?\s*$)',
    re.MULTILINE | re.DOTALL,
)
_ENV_GETENV = re.compile(
    r'os\s*\.\s*getenv\s*\(\s*["\'](?P<name>[A-Z][A-Z0-9_]*)["\']'
    r'(?:\s*,\s*(?P<default>"[^"]*"|\'[^\']*\'|None|True|False|-?\d+(?:\.\d+)?))?',
    re.DOTALL,
)


def _parse_env_default(body: str, name: str) -> Any:
    for match in _ENV_GETENV.finditer(body):
        if match.group("name") != name:
            continue
        default_raw = match.group("default")
        if default_raw is None:
            return None
        try:
            return ast.literal_eval(default_raw)
        except (SyntaxError, ValueError):
            return default_raw.strip()
    return None


def parse_vllm_ascend_env_vars(html: str) -> ParseResult:
    result = ParseResult(source_id="vllm-ascend-env-vars", kind="env_vars")
    for code in _extract_code_blocks(html) + [_strip_tags(html)]:
        for match in _ENV_ENTRY.finditer(code):
            name = match.group("name")
            if name in result.items:
                continue
            result.items[name] = {"upstream_default": _parse_env_default(match.group("body"), name)}
    if result.items:
        return result

    for table in _extract_tables(html):
        if not table or not table[0]:
            continue
        header = [cell.lower() for cell in table[0]]
        if not any("env" in cell or "name" in cell or "variable" in cell for cell in header):
            continue
        name_col = next((i for i, c in enumerate(header) if "name" in c or "variable" in c or "env" in c), 0)
        default_col = next((i for i, c in enumerate(header) if "default" in c), None)
        for row in table[1:]:
            if name_col >= len(row):
                continue
            raw_name = row[name_col].strip()
            name = re.sub(r"[^A-Z0-9_]", "", raw_name.upper())
            if not name:
                continue
            default = row[default_col].strip() if default_col is not None and default_col < len(row) else None
            result.items[name] = {"upstream_default": default or None}
    if not result.items:
        result.notes.append("no env vars code block or table detected")
    return result


# vllm-ascend supported models: tables with Model | Support | ... | Supported Hardware ...
def parse_vllm_ascend_supported_models(html: str) -> ParseResult:
    result = ParseResult(source_id="vllm-ascend-model-support", kind="supported_models")
    archs: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for table in _extract_tables(html):
        if not table or not table[0]:
            continue
        header = [cell.lower() for cell in table[0]]
        arch_col = next((i for i, c in enumerate(header) if "architecture" in c or "arch" in c), None)
        model_col = next((i for i, c in enumerate(header) if "model" in c and "type" not in c), None)
        support_col = next((i for i, c in enumerate(header) if c == "support"), None)
        hardware_col = next((i for i, c in enumerate(header) if "hardware" in c), None)
        if arch_col is None and model_col is None:
            continue
        for row in table[1:]:
            arch = row[arch_col].strip() if arch_col is not None and arch_col < len(row) else ""
            model = row[model_col].strip() if model_col is not None and model_col < len(row) else ""
            if not arch and not model:
                continue
            if model and model not in models:
                models[model] = {
                    "model": model,
                    "support": row[support_col].strip() if support_col is not None and support_col < len(row) else None,
                    "supported_hardware": row[hardware_col].strip() if hardware_col is not None and hardware_col < len(row) else None,
                }
            if not arch:
                continue
            arch_key = arch
            bucket = archs.setdefault(arch_key, {"architecture": arch_key, "models": []})
            if model and model not in bucket["models"]:
                bucket["models"].append(model)
    result.items = {"architectures": list(archs.values()), "models": list(models.values())}
    if not archs and not models:
        result.notes.append("no model table detected")
    return result


PARSERS: dict[str, Callable[[str], ParseResult]] = {
    "vllm-cli-serve-reference": parse_vllm_cli_serve,
    "vllm-ascend-env-vars": parse_vllm_ascend_env_vars,
    "vllm-ascend-model-support": parse_vllm_ascend_supported_models,
}


def _coerce_default(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text in {"None", "none", "null", ""}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# Differs
# ---------------------------------------------------------------------------

def diff_manifest_cli_params(
    upstream: dict[str, Any],
    local_manifest: dict[str, Any],
) -> list[DiffEntry]:
    """Compare upstream CLI params against local manifests/<engine>/<version>.yaml."""
    local = (local_manifest or {}).get("cli_params") or {}
    entries: list[DiffEntry] = []
    for name, meta in upstream.items():
        if name not in local:
            entries.append(DiffEntry(field=name, status="upstream_only", upstream=meta))
            continue
        upstream_default = meta.get("upstream_default")
        local_default = local[name].get("upstream_default")
        if _normalize_default(upstream_default) != _normalize_default(local_default):
            entries.append(DiffEntry(
                field=name,
                status="value_drift",
                upstream=upstream_default,
                local=local_default,
            ))
    for name in local:
        if name not in upstream:
            entries.append(DiffEntry(field=name, status="local_only", local=local[name]))
    return entries


def diff_env_vars(
    upstream: dict[str, Any],
    local_manifest: dict[str, Any],
) -> list[DiffEntry]:
    local = (local_manifest or {}).get("env_vars") or {}
    entries: list[DiffEntry] = []
    for name, meta in upstream.items():
        if name not in local:
            entries.append(DiffEntry(field=name, status="upstream_only", upstream=meta))
            continue
        upstream_default = meta.get("upstream_default")
        local_default = local[name].get("upstream_default")
        if _normalize_default(upstream_default) != _normalize_default(local_default):
            entries.append(DiffEntry(
                field=name,
                status="value_drift",
                upstream=upstream_default,
                local=local_default,
            ))
    for name in local:
        if name not in upstream:
            entries.append(DiffEntry(field=name, status="local_only", local=local[name]))
    return entries


def diff_supported_models(
    upstream: dict[str, Any],
    local_inventory: dict[str, Any],
) -> list[DiffEntry]:
    """Compare upstream supported architectures against models_inventory.yaml."""
    local_archs: set[str] = set()
    local_models: dict[str, str] = {}
    for entry in (local_inventory or {}).get("inventory") or []:
        if isinstance(entry, dict) and entry.get("architecture"):
            local_archs.add(str(entry["architecture"]))
        if isinstance(entry, dict):
            for model in entry.get("models") or []:
                local_models[_normalize_model_name(str(model))] = str(model)
    upstream_models = {
        _normalize_model_name(str(item.get("model"))): str(item.get("model"))
        for item in upstream.get("models") or []
        if isinstance(item, dict) and item.get("model")
    }
    if upstream_models:
        entries: list[DiffEntry] = []
        for key in sorted(upstream_models.keys() - local_models.keys()):
            entries.append(DiffEntry(field=upstream_models[key], status="upstream_only"))
        for key in sorted(local_models.keys() - upstream_models.keys()):
            entries.append(DiffEntry(field=local_models[key], status="local_only"))
        return entries

    upstream_archs = {
        str(item.get("architecture"))
        for item in upstream.get("architectures") or []
        if isinstance(item, dict) and item.get("architecture")
    }
    entries: list[DiffEntry] = []
    for arch in sorted(upstream_archs - local_archs):
        entries.append(DiffEntry(field=arch, status="upstream_only"))
    for arch in sorted(local_archs - upstream_archs):
        entries.append(DiffEntry(field=arch, status="local_only"))
    return entries


def _normalize_default(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def _normalize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    engine: str,
    source_id: str,
    fetch: FetchResult,
    parse: ParseResult | None,
    diff_entries: list[DiffEntry],
    target_files: list[str],
) -> Path:
    """Write a single source's report. Path: <output>/<engine>/<source_id>.json."""
    engine_dir = output_dir / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / f"{source_id}.json"
    payload = {
        "schema_version": 1,
        "engine": engine,
        "source_id": source_id,
        "source_url": fetch.url,
        "fetch": {
            "from_cache": fetch.from_cache,
            "error": fetch.error,
            "body_bytes": len(fetch.body),
        },
        "parse": {
            "kind": parse.kind if parse else None,
            "item_count": _parse_item_count(parse),
            "notes": parse.notes if parse else [],
        },
        "target_files": target_files,
        "summary": {
            "upstream_only": sum(1 for d in diff_entries if d.status == "upstream_only"),
            "local_only": sum(1 for d in diff_entries if d.status == "local_only"),
            "value_drift": sum(1 for d in diff_entries if d.status == "value_drift"),
        },
        "diff": [
            {
                "field": d.field,
                "status": d.status,
                **({"upstream": d.upstream} if d.upstream is not None else {}),
                **({"local": d.local} if d.local is not None else {}),
            }
            for d in diff_entries
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


def _parse_item_count(parse: ParseResult | None) -> int:
    if parse is None:
        return 0
    if parse.kind == "supported_models" and isinstance(parse.items.get("models"), list):
        return len(parse.items["models"])
    return len(parse.items)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    repo_root: Path
    config_root: Path
    output_dir: Path
    cache_dir: Path
    fixture_dir: Path | None
    use_cache: bool
    engines: list[str]


def _engine_manifest_path(config_root: Path, engine: str) -> Path | None:
    base = config_root / "manifests" / engine
    if not base.is_dir():
        return None
    files = sorted(p for p in base.glob("*.yaml") if p.is_file())
    return files[-1] if files else None


def run_sync(cfg: SyncConfig) -> dict[str, Any]:
    """Run the full sync pipeline and return the aggregate report."""
    sources_catalog = load_structured_file(cfg.config_root / "official_sources.yaml")
    inventory_path = cfg.config_root / "models_inventory.yaml"
    inventory = load_structured_file(inventory_path) if inventory_path.is_file() else {}

    engines_data = sources_catalog.get("engines") or {}
    aggregate: dict[str, Any] = {"engines": {}, "summary": {"reports": 0, "fetch_errors": 0, "drift_total": 0}}

    for engine in cfg.engines:
        engine_data = engines_data.get(engine)
        if not isinstance(engine_data, dict):
            continue
        target_files = engine_data.get("target_config_files") or []
        manifest_path = _engine_manifest_path(cfg.config_root, engine)
        local_manifest = load_structured_file(manifest_path) if manifest_path else {}

        engine_summary: list[dict[str, Any]] = []
        for source in engine_data.get("sources") or []:
            source_id = source.get("id")
            parser = PARSERS.get(source_id)
            if not parser:
                continue
            fetch = fetch_url(
                source_id,
                source["url"],
                cache_dir=cfg.cache_dir,
                use_cache=cfg.use_cache,
                fixture_dir=cfg.fixture_dir,
            )
            parse = None
            diff_entries: list[DiffEntry] = []
            if fetch.error:
                aggregate["summary"]["fetch_errors"] += 1
            else:
                parse = parser(fetch.body)
                diff_entries = _route_diff(source_id, parse, local_manifest, inventory)
            report_path = write_report(
                cfg.output_dir, engine, source_id, fetch, parse, diff_entries, target_files,
            )
            aggregate["summary"]["reports"] += 1
            aggregate["summary"]["drift_total"] += len(diff_entries)
            engine_summary.append({
                "source_id": source_id,
                "report": str(report_path.relative_to(cfg.repo_root)) if report_path.is_relative_to(cfg.repo_root) else str(report_path),
                "drift_count": len(diff_entries),
                "fetch_error": fetch.error,
            })
        aggregate["engines"][engine] = engine_summary
    return aggregate


def _route_diff(
    source_id: str,
    parse: ParseResult,
    local_manifest: dict[str, Any],
    inventory: dict[str, Any],
) -> list[DiffEntry]:
    if parse.kind == "cli_params":
        return diff_manifest_cli_params(parse.items, local_manifest)
    if parse.kind == "env_vars":
        return diff_env_vars(parse.items, local_manifest)
    if parse.kind == "supported_models":
        return diff_supported_models(parse.items, inventory)
    return []
