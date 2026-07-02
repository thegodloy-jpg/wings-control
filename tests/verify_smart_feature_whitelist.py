# -*- coding: utf-8 -*-
"""Audit smart_feature_whitelist.json against the requirement matrix.

This verifier is intentionally stricter than the normal dry-run coverage:
it checks the JSON table itself for missing/extra rows, then runs every
whitelisted tuple through the production dry-run pipeline with all three
feature switches set by env.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
WINGS = ROOT / "wings_control"
OUT_PATH = TESTS / "smart_feature_whitelist_audit_output.txt"

sys.path.insert(0, str(WINGS))
sys.path.insert(0, str(TESTS))

from utils.model_utils import resolve_feature_whitelist  # noqa: E402
from _dryrun_req_harness import exec_line, run_case  # noqa: E402
from core.wings_entry import _resolve_lmcache_install_target  # noqa: E402


@dataclass(frozen=True)
class Expect:
    key: str
    engine: str
    names: tuple[str, ...]
    cards: tuple[str, ...]
    features: frozenset[str]
    source: str
    model_name: str
    arch: str
    card_env: dict[str, str]
    quant_method: str | None = None
    spec_variant: str | None = None
    sparse_variant: str | None = None
    offload_variant: str | None = None


EXPECTED: tuple[Expect, ...] = (
    Expect("vllm:qwen35", "vllm", ("qwen3.5-397b", "qwen3_5-397b"), ("*",), frozenset({"spec", "sparse"}), "0430",
           "Qwen3.5-397B-A17B", "Qwen3_5MoeForConditionalGeneration",
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"}, spec_variant="qwen3_5_mtp", sparse_variant="fp8"),
    Expect("vllm:glm47", "vllm", ("glm-4.7",), ("*",), frozenset({"spec", "sparse", "offload"}), "0430",
           "glm-4.7", "Glm4MoeForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"},
           spec_variant="suffix", sparse_variant="fp8", offload_variant="lmcache_cpu+auto"),
    Expect("vllm:glm51", "vllm", ("glm-5.1", "glm5.1"), ("*",), frozenset({"sparse"}), "0430",
           "glm-5.1", "GlmMoeDsaForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"},
           spec_variant="suffix", sparse_variant="indexcache_topk4"),
    Expect("vllm:minimax27", "vllm", ("minimax-m2.7", "minimax-m27"), ("*",), frozenset({"spec", "sparse", "offload"}), "0430",
           "MiniMax-M2.7", "MiniMaxM2ForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"},
           spec_variant="suffix", sparse_variant="fp8", offload_variant="lmcache_cpu+auto"),
    Expect("vllm:v4flash", "vllm", ("deepseek-v4-flash", "v4-flash"), ("*",), frozenset({"spec", "sparse", "offload"}), "day0",
           "DeepSeek-V4-Flash", "DeepseekV4ForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp"},
           spec_variant="deepseek_mtp", sparse_variant="indexcache_use_index_cache_topk4",
           offload_variant="native_kv_offloading_backend"),
    Expect("asc:glm47", "vllm_ascend", ("glm-4.7",), ("910b", "910c"), frozenset({"spec", "offload"}), "0430",
           "glm-4.7", "Glm4MoeForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="suffix", offload_variant="lmcache_cpu+auto"),
    Expect("asc:minimax25", "vllm_ascend", ("minimax-m2.5", "minimax-m25"), ("910b", "910c"), frozenset({"spec", "offload"}), "0430",
           "MiniMax-M2.5-w8a8", "MiniMaxM2ForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="suffix", offload_variant="lmcache_cpu+auto"),
    Expect("asc:deepseek32", "vllm_ascend", ("deepseek-v3.2", "deepseek_v3.2"), ("910c",), frozenset({"spec", "offload"}), "0430",
           "DeepSeek-V3.2", "DeepseekV32ForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="suffix", offload_variant="lmcache_cpu+auto"),
    Expect("asc:qwen36", "vllm_ascend", ("qwen3.6", "qwen3_6"), ("910c",), frozenset({"spec"}), "0430",
           "Qwen3.6-35B-A3B", "Qwen3_5MoeForConditionalGeneration",
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="qwen3_5_mtp"),
    Expect("asc:v4flash", "vllm_ascend", ("deepseek-v4-flash", "v4-flash"), ("910b", "910c"), frozenset({"spec", "sparse", "offload"}), "day0",
           "DeepSeek-V4-Flash", "DeepseekV4ForCausalLM",
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="deepseek_mtp", sparse_variant="noop",
           offload_variant="native_cpu_connector"),
    Expect("asc:glm51", "vllm_ascend", ("glm-5.1", "glm5.1"), ("910b", "910c"), frozenset({"sparse"}), "day0",
           "glm-5.1", "GlmMoeDsaForCausalLM", {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="suffix", sparse_variant="indexcache_topk8"),
    Expect("asc:glm52", "vllm_ascend", ("glm-5.2", "glm5.2"), ("910b", "910c"), frozenset({"spec"}), "day0",
           "GLM-5.2-w8a8", "GlmMoeDsaForCausalLM",
           {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910c"},
           "ascend", spec_variant="deepseek_mtp"),
)

NEGATIVE_CASES = (
    ("qwen3.6 must not match 910b", "vllm_ascend", "Qwen3.6-35B-A3B", "/models/Qwen3.6-35B-A3B", "ascend910b", frozenset()),
    ("deepseek-v3.2 must not match 910b", "vllm_ascend", "DeepSeek-V3.2", "/models/DeepSeek-V3.2", "ascend910b", frozenset()),
    ("qwen3.6 must not match vllm/nv", "vllm", "Qwen3.6-35B-A3B", "/models/Qwen3.6-35B-A3B", "", frozenset()),
    ("glm-5 base must not match glm-5.1/5.2", "vllm", "glm-5", "/models/glm-5", "", frozenset()),
    ("v4-pro must not match v4-flash", "vllm_ascend", "DeepSeek-V4-Pro", "/models/DeepSeek-V4-Pro", "ascend910c", frozenset()),
    ("ascend empty card must miss", "vllm_ascend", "glm-4.7", "/models/glm-4.7", "", frozenset()),
)


@contextmanager
def _patched_env(values: dict[str, str]):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _row_key(row: dict) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        str(row.get("engine", "")),
        tuple(str(x).lower() for x in row.get("name_tokens", ())),
        tuple(str(x).lower() for x in row.get("card_tokens", ())),
    )


def _load_json() -> dict:
    path = WINGS / "config" / "smart_feature_whitelist.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_json_shape(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["whitelist root is not a JSON object"]
    expected_by_key = {
        (e.engine, e.names, e.cards): e for e in EXPECTED
    }
    actual_by_key: dict[tuple[str, tuple[str, ...], tuple[str, ...]], set[str]] = {}
    for feat in ("spec", "sparse", "offload"):
        rows = data.get(feat)
        if not isinstance(rows, list):
            errors.append(f"{feat}: table is missing or not a list")
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{feat}[{i}]: row is not an object")
                continue
            key = _row_key(row)
            actual_by_key.setdefault(key, set()).add(feat)
            if key not in expected_by_key:
                errors.append(f"{feat}[{i}]: unexpected row engine/names/cards={key}")
                continue
            exp = expected_by_key[key]
            if feat not in exp.features:
                errors.append(f"{feat}[{i}]: feature not expected for {exp.key}")
            if row.get("source") != exp.source:
                errors.append(f"{feat}[{i}]: source {row.get('source')!r} != {exp.source!r} for {exp.key}")
            if feat == "sparse":
                topk = row.get("topk")
                if not isinstance(topk, dict) or "accuracy_first" not in topk:
                    errors.append(f"{feat}[{i}]: sparse row missing topk.accuracy_first for {exp.key}")

    for e in EXPECTED:
        got = actual_by_key.get((e.engine, e.names, e.cards), set())
        if got != set(e.features):
            errors.append(f"{e.key}: feature set mismatch, got={sorted(got)} expected={sorted(e.features)}")
    return errors


def _check_resolver() -> list[str]:
    errors: list[str] = []
    for e in EXPECTED:
        for name in e.names:
            for card in e.cards:
                card_value = "" if card == "*" else f"ascend{card}"
                got = resolve_feature_whitelist(e.engine, name, f"/models/{name}", card_value)
                if got != e.features:
                    errors.append(f"{e.key}: resolver({name},{card_value or '*'}) got={sorted(got)} expected={sorted(e.features)}")
    for label, engine, name, path, card, expected in NEGATIVE_CASES:
        got = resolve_feature_whitelist(engine, name, path, card)
        if got != expected:
            errors.append(f"{label}: got={sorted(got)} expected={sorted(expected)}")
    return errors


def _model_config(e: Expect) -> dict:
    cfg = {"architecture": e.arch}
    if e.quant_method:
        cfg["quantization_config"] = {"quant_method": e.quant_method}
    return cfg


def _check_dryrun() -> list[str]:
    errors: list[str] = []
    base_env = {
        "ENABLE_SPECULATIVE_DECODE": "true",
        "ENABLE_SPARSE": "true",
        "ENABLE_KV_OFFLOAD": "true",
        "ENABLE_KV_MEM_OFFLOAD": "true",
        "KV_MEM_OFFLOAD_SIZE": "auto",
        "AVAILABLE_POD_MEM_SIZE": "512",
    }
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for e in EXPECTED:
            r = run_case(
                {"model-name": e.model_name, "engine": e.engine, "device-count": 8},
                {**e.card_env, **base_env},
                _model_config(e),
            )
            feats = r.features
            variants = r.variants
            if feats.get("sparse_kv") is not ("sparse" in e.features):
                errors.append(f"{e.key}: sparse_kv={feats.get('sparse_kv')} expected={'sparse' in e.features}")
            if feats.get("kv_offload") is not ("offload" in e.features):
                errors.append(f"{e.key}: kv_offload={feats.get('kv_offload')} expected={'offload' in e.features}")
            if e.spec_variant and variants.get("speculative_decode") != e.spec_variant:
                errors.append(f"{e.key}: spec variant={variants.get('speculative_decode')} expected={e.spec_variant}")
            if e.sparse_variant and variants.get("sparse_kv") != e.sparse_variant:
                errors.append(f"{e.key}: sparse variant={variants.get('sparse_kv')} expected={e.sparse_variant}")
            if e.offload_variant and variants.get("kv_offload") != e.offload_variant:
                errors.append(f"{e.key}: offload variant={variants.get('kv_offload')} expected={e.offload_variant}")
    finally:
        logging.disable(previous_disable)
    return errors


def _command_contains(command: str, needle: str) -> bool:
    return needle in command or needle in exec_line(command)


def _assert_not_contains(errors: list[str], label: str, command: str, needles: tuple[str, ...]) -> None:
    for needle in needles:
        if _command_contains(command, needle):
            errors.append(f"{label}: unexpected command fragment {needle!r}")


def _check_env_bypass() -> list[str]:
    """Prove user-provided env switches cannot bypass the smart whitelist."""
    errors: list[str] = []
    all_switches = {
        "ENABLE_SPECULATIVE_DECODE": "true",
        "SD_ENABLE": "true",
        "ENABLE_SPARSE": "true",
        "SPARSE_ENABLE": "true",
        "SPARSE_LEVEL": "performance_first",
        "ENABLE_KV_OFFLOAD": "true",
        "LMCACHE_OFFLOAD": "true",
        "ENABLE_KV_MEM_OFFLOAD": "true",
        "LMCACHE_LOCAL_CPU": "true",
        "KV_MEM_OFFLOAD_SIZE": "200",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "200",
        "ENABLE_KV_DISK_OFFLOAD": "true",
        "KV_DISK_OFFLOAD_PATH": "/tmp/kv",
        "KV_DISK_OFFLOAD_SIZE": "100",
        "ENABLE_KV_QAT": "true",
        "KV_QAT_COMPRESS_LEVEL": "1",
        "KV_QAT_INSTANCE_NUM": "1",
        "AVAILABLE_POD_MEM_SIZE": "512",
    }
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        cases = [
            (
                "env-all llama nv non-whitelist",
                {"model-name": "Llama-3-70B", "engine": "vllm", "device-count": 8},
                {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", **all_switches},
                {"architecture": "LlamaForCausalLM"},
                {"speculative_decode": True, "sparse_kv": False, "kv_offload": False},
                {"speculative_decode": "suffix", "sparse_kv": None, "kv_offload": None},
            ),
            (
                "env-all qwen36 ascend 910b non-whitelist",
                {"model-name": "Qwen3.6-35B-A3B", "engine": "vllm_ascend", "device-count": 8},
                {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "WINGS_DEVICE_NAME": "ascend910b", **all_switches},
                {"architecture": "Qwen3_5MoeForConditionalGeneration", "quantization_config": {"quant_method": "ascend"}},
                {"speculative_decode": True, "sparse_kv": False, "kv_offload": False},
                {"speculative_decode": "suffix", "sparse_kv": None, "kv_offload": None},
            ),
            (
                "env-all glm51 nv sparse-only",
                {"model-name": "glm-5.1", "engine": "vllm", "device-count": 8},
                {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", **all_switches},
                {"architecture": "GlmMoeDsaForCausalLM"},
                {"speculative_decode": True, "sparse_kv": True, "kv_offload": False},
                {"speculative_decode": "suffix", "sparse_kv": "indexcache_topk4", "kv_offload": None},
            ),
            (
                "env-all qwen35 nv spec-sparse",
                {"model-name": "Qwen3.5-397B-A17B", "engine": "vllm", "device-count": 8},
                {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", **all_switches},
                {"architecture": "Qwen3_5MoeForConditionalGeneration"},
                {"speculative_decode": True, "sparse_kv": True, "kv_offload": False},
                {"speculative_decode": "qwen3_5_mtp", "sparse_kv": "fp8", "kv_offload": None},
            ),
            (
                "env-all pd veto",
                {"model-name": "glm-4.7", "engine": "vllm", "device-count": 8},
                {"DISTRIBUTED_EXECUTOR_BACKEND": "mp", "PD_ROLE": "P", **all_switches},
                {"architecture": "Glm4MoeForCausalLM"},
                {"speculative_decode": False, "sparse_kv": False, "kv_offload": False},
                {"speculative_decode": None, "sparse_kv": None, "kv_offload": None},
            ),
        ]
        for label, cli, env, cfg, expected_features, expected_variants in cases:
            r = run_case(cli, env, cfg)
            for key, expected in expected_features.items():
                if r.features.get(key) is not expected:
                    errors.append(f"{label}: features.{key}={r.features.get(key)} expected={expected}")
            for key, expected in expected_variants.items():
                if r.variants.get(key) != expected:
                    errors.append(f"{label}: variants.{key}={r.variants.get(key)!r} expected={expected!r}")

            if not expected_features["kv_offload"]:
                _assert_not_contains(
                    errors,
                    label,
                    r.command,
                    (
                        "export ENABLE_KV_OFFLOAD=true",
                        "export ENABLE_KV_MEM_OFFLOAD=true",
                        "export KV_MEM_OFFLOAD_SIZE=",
                        "export LMCACHE_CONFIG_FILE=",
                        "LMCacheConnectorV1",
                        "CPUOffloadingConnector",
                        "--kv-offloading-backend",
                        "--lmcache-target",
                    ),
                )
            if not expected_features["sparse_kv"]:
                _assert_not_contains(
                    errors,
                    label,
                    r.command,
                    ("--hf-overrides", "--kv-cache-dtype fp8", "--calculate-kv-scales", "indexcache"),
                )

        with _patched_env({"ENABLE_KV_OFFLOAD": "true"}):
            target = _resolve_lmcache_install_target(
                "vllm",
                {
                    "engine": "vllm",
                    "model_name": "Llama-3-70B",
                    "model_path": "/models/Llama-3-70B",
                    "_smart_feats": [],
                },
            )
        if target is not None:
            errors.append(f"lmcache install second guard: target={target!r} expected=None")
    finally:
        logging.disable(previous_disable)
    return errors


def main() -> int:
    sections: list[tuple[str, list[str]]] = []
    data = _load_json()
    sections.append(("JSON schema and exact matrix", _check_json_shape(data)))
    sections.append(("Resolver aliases/cards/negative cases", _check_resolver()))
    sections.append(("Production dry-run variants", _check_dryrun()))
    sections.append(("Env bypass negative cases", _check_env_bypass()))

    lines = [
        "=" * 96,
        "Smart feature whitelist audit",
        "=" * 96,
        f"Expected unique tuples: {len(EXPECTED)}",
        "Expected split rows: spec=10 sparse=7 offload=7",
        "",
    ]
    failed = 0
    for name, errors in sections:
        status = "PASS" if not errors else "FAIL"
        failed += len(errors)
        lines.append(f"[{status}] {name}: {len(errors)} issue(s)")
        for err in errors:
            lines.append(f"  - {err}")
        lines.append("")
    lines.append("=" * 96)
    lines.append(f"RESULT: {'PASS' if failed == 0 else 'FAIL'}  issues={failed}")
    lines.append("=" * 96)

    text = "\n".join(lines) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
