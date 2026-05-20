#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only coverage report for supported models, recipes, and official sources.

This tool intentionally does not crawl remote documentation and does not write
project config.  It answers whether the models recognized by model_utils have a
local recipe and at least one candidate source in official_sources.yaml.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from _config_tooling import load_structured_file

_REPO_ROOT = Path(__file__).resolve().parent.parent

_TABLE_BY_MODEL_TYPE = {
    "llm": "_LLM_MODELS",
    "embedding": "_EMBEDDING_MODELS",
    "rerank": "_RERANK_MODELS",
}
_MODEL_TYPE_ORDER = {"llm": 0, "embedding": 1, "rerank": 2}

_FAMILY_MARKERS = {
    "deepseek": ("deepseek", "deepseekv3", "deepseekv32"),
    "qwen": ("qwen", "qwq"),
    "glm": ("glm",),
    "kimi": ("kimi", "k25"),
    "minimax": ("minimax",),
    "llama": ("llama", "meta-llama"),
    "bge": ("bge", "xlmroberta", "bertmodel"),
}
_GENERIC_LLM_SCENARIOS = {
    "openai_compatible_server",
    "single_node",
    "text_generation_service",
    "tool_calling",
    "reasoning",
}
_TASK_SCENARIOS = {
    "embedding": {"embedding", "pooling"},
    "rerank": {"rerank", "reranker"},
}


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _load_models_inventory(
    inventory_path: Path,
) -> dict[str, dict[str, list[str]]]:
    data = load_structured_file(inventory_path)
    result: dict[str, dict[str, list[str]]] = {
        "_LLM_MODELS": {},
        "_EMBEDDING_MODELS": {},
        "_RERANK_MODELS": {},
    }
    for entry in data.get("inventory") or []:
        if not isinstance(entry, dict):
            continue
        model_type = str(entry.get("type") or "")
        table_name = _TABLE_BY_MODEL_TYPE.get(model_type)
        architecture = entry.get("architecture")
        if table_name is None or not architecture:
            continue
        result[table_name][str(architecture)] = _as_string_list(entry.get("models"))
    return result


def _literal_model_tables(
    module: ast.Module,
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    wanted = set(_TABLE_BY_MODEL_TYPE.values())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            value = ast.literal_eval(statement.value)
            if not isinstance(value, dict):
                continue
            result[target.id] = {
                str(architecture): _as_string_list(models)
                for architecture, models in value.items()
            }
    return result


def _uses_inventory_loader(module: ast.Module) -> bool:
    wanted = tuple(_TABLE_BY_MODEL_TYPE.values())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if not isinstance(target, ast.Tuple):
                continue
            names = tuple(
                item.id for item in target.elts if isinstance(item, ast.Name)
            )
            if names[:3] != wanted:
                continue
            if isinstance(statement.value, ast.Call):
                func = statement.value.func
                return isinstance(func, ast.Name) and func.id == "_load_models_inventory"
    return False


def load_supported_model_tables(
    model_utils_path: str | Path,
) -> dict[str, dict[str, list[str]]]:
    """Load the three model tables from model_utils.py.

    Current branches hydrate the tables from config/models_inventory.yaml.
    Older branches kept literal dictionaries in model_utils.py.  This function
    supports both shapes so the report stays tied to model_utils as the entry
    point instead of hard-coding one carrier.
    """
    path = Path(model_utils_path)
    module = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    literal_tables = _literal_model_tables(module)
    if set(literal_tables) == set(_TABLE_BY_MODEL_TYPE.values()):
        return literal_tables

    if _uses_inventory_loader(module):
        inventory_path = path.resolve().parents[1] / "config" / "models_inventory.yaml"
        return _load_models_inventory(inventory_path)

    missing = sorted(set(_TABLE_BY_MODEL_TYPE.values()) - set(literal_tables))
    raise ValueError(f"failed to resolve model table(s) from {path}: {missing}")


def iter_supported_models(
    tables: dict[str, dict[str, list[str]]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for model_type, table_name in _TABLE_BY_MODEL_TYPE.items():
        for architecture, model_names in tables.get(table_name, {}).items():
            for model_name in model_names:
                records.append({
                    "model_name": str(model_name),
                    "model_type": model_type,
                    "architecture": str(architecture),
                })
    return sorted(
        records,
        key=lambda item: (
            _MODEL_TYPE_ORDER.get(item["model_type"], 99),
            item["architecture"].lower(),
            item["model_name"].lower(),
        ),
    )


def _iter_config_files_recursive(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml", "*.json"):
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files))


def _recipe_ref(path: Path, kind: str, repo_root: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": _rel(path, repo_root),
        "experimental": "_experimental" in path.parts,
    }


def _append_unique_ref(
    bucket: dict[str, list[dict[str, Any]]],
    key: str,
    ref: dict[str, Any],
) -> None:
    refs = bucket.setdefault(key, [])
    if ref not in refs:
        refs.append(ref)


def _collect_recipe_catalog(
    config_root: Path,
    repo_root: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        "models": {},
        "architectures": {},
    }

    for path in _iter_config_files_recursive(config_root / "recipes" / "models"):
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        names: set[str] = set()
        model_id = data.get("model_id")
        if model_id:
            names.add(str(model_id))
        matches = data.get("matches") or {}
        if isinstance(matches, dict):
            names.update(_as_string_list(matches.get("model_names")))
            names.update(_as_string_list(matches.get("model_paths")))
        names.add(path.stem)
        ref = _recipe_ref(path, "model", repo_root)
        for name in names:
            _append_unique_ref(result["models"], name.lower(), ref)

    for path in _iter_config_files_recursive(config_root / "recipes" / "architectures"):
        try:
            data = load_structured_file(path)
        except Exception:
            continue
        names: set[str] = set()
        architecture = data.get("architecture")
        if architecture:
            names.add(str(architecture))
        names.update(_as_string_list(data.get("also_matches")))
        ref = _recipe_ref(path, "architecture", repo_root)
        for name in names:
            _append_unique_ref(result["architectures"], name, ref)

    return result


def _recipe_coverage(
    model: dict[str, str],
    recipe_catalog: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[str, list[dict[str, Any]]]:
    model_refs = recipe_catalog["models"].get(model["model_name"].lower(), [])
    arch_refs = recipe_catalog["architectures"].get(model["architecture"], [])
    refs = sorted(
        model_refs + arch_refs,
        key=lambda item: (item["kind"], item["path"]),
    )
    if model_refs:
        return "model_recipe", refs
    if arch_refs:
        return "architecture_recipe", refs
    return "missing", []


def _normal_text(value: str) -> str:
    return value.lower().replace("_", "-")


def _infer_model_terms(model: dict[str, str]) -> list[str]:
    text = _normal_text(f"{model['model_name']} {model['architecture']}")
    terms: set[str] = set()
    for family, markers in _FAMILY_MARKERS.items():
        if any(_normal_text(marker) in text for marker in markers):
            terms.add(family)
    terms.update(_TASK_SCENARIOS.get(model["model_type"], set()))
    if model["model_type"] == "embedding" and "qwen" in terms:
        terms.add("qwen_embedding")
    if model["model_type"] == "rerank" and "qwen" in terms:
        terms.add("qwen_reranker")
    return sorted(terms)


def _catalog_text(engine: str, data: dict[str, Any]) -> str:
    searchable = {
        "engine": engine,
        "scenario_groups": data.get("scenario_groups") or [],
        "sources": [
            {
                "id": source.get("id"),
                "url": source.get("url"),
                "extract_targets": source.get("extract_targets"),
                "target_updates": source.get("target_updates"),
            }
            for source in data.get("sources") or []
            if isinstance(source, dict)
        ],
    }
    return _normal_text(json.dumps(searchable, ensure_ascii=False))


def _source_ids(data: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for source in data.get("sources") or []:
        if isinstance(source, dict) and source.get("id"):
            ids.append(str(source["id"]))
    return ids


def _source_candidates(
    model: dict[str, str],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    engines = catalog.get("engines") or {}
    if not isinstance(engines, dict):
        return []

    model_terms = _infer_model_terms(model)
    for engine, raw_data in sorted(engines.items()):
        if not isinstance(raw_data, dict) or raw_data.get("enabled") is False:
            continue
        text = _catalog_text(str(engine), raw_data)
        scenarios = {_normal_text(str(item)) for item in raw_data.get("scenario_groups") or []}
        matched_terms = [term for term in model_terms if _normal_text(term) in text]
        confidence = "family_or_task" if matched_terms else ""

        if not matched_terms and model["model_type"] == "llm":
            generic = sorted(scenarios & {_normal_text(term) for term in _GENERIC_LLM_SCENARIOS})
            if generic:
                matched_terms = generic
                confidence = "generic_llm"
        if not matched_terms and model["model_type"] in _TASK_SCENARIOS:
            task_terms = {_normal_text(term) for term in _TASK_SCENARIOS[model["model_type"]]}
            task_matches = sorted(scenarios & task_terms)
            if task_matches:
                matched_terms = task_matches
                confidence = "task"

        if matched_terms:
            candidates.append({
                "engine": str(engine),
                "confidence": confidence,
                "match_terms": matched_terms,
                "source_ids": _source_ids(raw_data),
            })
    return candidates


def build_coverage_report(
    repo_root: str | Path = _REPO_ROOT,
    *,
    model_utils_path: str | Path | None = None,
    source_catalog_path: str | Path | None = None,
    config_root: str | Path | None = None,
    model_name_filter: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = Path(config_root).resolve() if config_root else repo / "wings_control" / "config"
    model_utils = (
        Path(model_utils_path).resolve()
        if model_utils_path
        else repo / "wings_control" / "utils" / "model_utils.py"
    )
    sources_path = (
        Path(source_catalog_path).resolve()
        if source_catalog_path
        else config / "official_sources.yaml"
    )

    tables = load_supported_model_tables(model_utils)
    models = iter_supported_models(tables)
    if model_name_filter:
        needle = model_name_filter.lower()
        models = [model for model in models if needle in model["model_name"].lower()]

    recipe_catalog = _collect_recipe_catalog(config, repo)
    source_catalog = load_structured_file(sources_path)

    report_models: list[dict[str, Any]] = []
    for model in models:
        recipe_status, recipe_refs = _recipe_coverage(model, recipe_catalog)
        candidates = _source_candidates(model, source_catalog)
        gaps: list[str] = []
        if recipe_status == "missing":
            gaps.append("missing_recipe")
        if not candidates:
            gaps.append("no_official_source_candidate")
        report_models.append({
            **model,
            "recipe_status": recipe_status,
            "recipe_refs": recipe_refs,
            "source_candidates": candidates,
            "gaps": gaps,
        })

    summary = {
        "models_total": len(report_models),
        "model_recipe_count": sum(
            1 for item in report_models if item["recipe_status"] == "model_recipe"
        ),
        "architecture_recipe_count": sum(
            1 for item in report_models if item["recipe_status"] == "architecture_recipe"
        ),
        "missing_recipe_count": sum(
            1 for item in report_models if item["recipe_status"] == "missing"
        ),
        "with_source_candidate_count": sum(
            1 for item in report_models if item["source_candidates"]
        ),
        "gap_models_count": sum(1 for item in report_models if item["gaps"]),
    }

    return {
        "schema_version": 1,
        "model_utils": _rel(model_utils, repo),
        "official_sources": _rel(sources_path, repo),
        "summary": summary,
        "models": report_models,
    }


def format_json_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)


def format_text_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Official model coverage report",
        f"model_utils: {report['model_utils']}",
        f"official_sources: {report['official_sources']}",
        (
            "summary: "
            f"models_total={summary['models_total']}, "
            f"model_recipe={summary['model_recipe_count']}, "
            f"architecture_recipe={summary['architecture_recipe_count']}, "
            f"missing_recipe={summary['missing_recipe_count']}, "
            f"with_source_candidate={summary['with_source_candidate_count']}, "
            f"gap_models={summary['gap_models_count']}"
        ),
        "",
    ]
    for model in report["models"]:
        sources = ",".join(
            candidate["engine"] for candidate in model["source_candidates"]
        ) or "-"
        refs = ",".join(ref["path"] for ref in model["recipe_refs"]) or "-"
        gaps = ",".join(model["gaps"]) or "-"
        lines.append(
            f"- [{model['model_type']}] {model['model_name']} | "
            f"arch={model['architecture']} | "
            f"recipe={model['recipe_status']} | "
            f"sources={sources} | gaps={gaps} | refs={refs}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only report: model_utils supported models vs local recipes "
            "and official_sources.yaml candidates."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO_ROOT),
        help="Repository root. Defaults to the parent of tools/.",
    )
    parser.add_argument(
        "--model-utils",
        default=None,
        help="Override path to wings_control/utils/model_utils.py.",
    )
    parser.add_argument(
        "--config-root",
        default=None,
        help="Override path to wings_control/config/.",
    )
    parser.add_argument(
        "--source-catalog",
        default=None,
        help="Override path to official_sources.yaml.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional case-insensitive substring filter for model names.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Return exit code 1 when any selected model has a gap.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_coverage_report(
        args.repo_root,
        model_utils_path=args.model_utils,
        source_catalog_path=args.source_catalog,
        config_root=args.config_root,
        model_name_filter=args.model_name,
    )
    if args.output_json:
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    if args.fail_on_gap and report["summary"]["gap_models_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
