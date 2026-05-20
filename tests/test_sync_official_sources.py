# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""Offline tests for tools/sync_official_sources.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from _sync_official import (  # noqa: E402
    SyncConfig,
    diff_env_vars,
    diff_manifest_cli_params,
    diff_supported_models,
    fetch_url,
    parse_vllm_ascend_env_vars,
    parse_vllm_ascend_supported_models,
    parse_vllm_cli_serve,
    run_sync,
)


FIXTURES = ROOT / "tests" / "fixtures" / "official_sources_html"
CONFIG_ROOT = ROOT / "wings_control" / "config"


class TestParsers(unittest.TestCase):
    def test_vllm_cli_serve_extracts_options_and_defaults(self):
        html = (FIXTURES / "vllm-cli-serve-reference.html").read_text(encoding="utf-8")
        result = parse_vllm_cli_serve(html)
        self.assertEqual(result.kind, "cli_params")
        self.assertIn("model", result.items)
        self.assertIsNone(result.items["host"]["upstream_default"])
        self.assertEqual(result.items["port"]["upstream_default"], 8000)
        self.assertEqual(result.items["gpu_memory_utilization"]["upstream_default"], 0.92)
        self.assertIsNone(result.items["max_model_len"]["upstream_default"])
        self.assertTrue(result.items["allow_deprecated_quantization"]["deprecated"])

    def test_vllm_ascend_env_vars_parses_table(self):
        html = (FIXTURES / "vllm-ascend-env-vars.html").read_text(encoding="utf-8")
        result = parse_vllm_ascend_env_vars(html)
        self.assertEqual(result.kind, "env_vars")
        self.assertIn("VLLM_ASCEND_ENABLE_MLAPO", result.items)
        self.assertEqual(result.items["VLLM_ASCEND_ENABLE_MLAPO"]["upstream_default"], "1")
        self.assertIn("ASCEND_HOME_PATH", result.items)

    def test_vllm_ascend_supported_models_extracts_arch_list(self):
        html = (FIXTURES / "vllm-ascend-model-support.html").read_text(encoding="utf-8")
        result = parse_vllm_ascend_supported_models(html)
        self.assertEqual(result.kind, "supported_models")
        models = {item["model"] for item in result.items["models"]}
        self.assertIn("Qwen3", models)
        self.assertIn("DeepSeek V3/3.1", models)
        self.assertIn("Qwen3-Embedding", models)

    def test_parsers_tolerate_unrelated_html(self):
        result = parse_vllm_cli_serve("<html><body>nothing useful</body></html>")
        self.assertEqual(result.items, {})
        self.assertTrue(result.notes)


class TestDiffers(unittest.TestCase):
    def test_diff_manifest_detects_drift_and_new_flags(self):
        upstream = {
            "host": {"upstream_default": "0.0.0.0"},
            "port": {"upstream_default": 8000},
            "gpu_memory_utilization": {"upstream_default": 0.95},
            "new_flag": {"upstream_default": None},
        }
        local = {
            "cli_params": {
                "host": {"upstream_default": "0.0.0.0"},
                "port": {"upstream_default": 8000},
                "gpu_memory_utilization": {"upstream_default": 0.9},
                "legacy_flag": {"upstream_default": None},
            }
        }
        entries = diff_manifest_cli_params(upstream, local)
        statuses = {(e.field, e.status) for e in entries}
        self.assertIn(("new_flag", "upstream_only"), statuses)
        self.assertIn(("legacy_flag", "local_only"), statuses)
        self.assertIn(("gpu_memory_utilization", "value_drift"), statuses)

    def test_diff_env_vars_symmetric(self):
        entries = diff_env_vars(
            {"NEW": {"upstream_default": None}, "SHARED": {"upstream_default": "x"}},
            {"env_vars": {"SHARED": {"upstream_default": "x"}, "OLD": {"upstream_default": None}}},
        )
        statuses = {(e.field, e.status) for e in entries}
        self.assertIn(("NEW", "upstream_only"), statuses)
        self.assertIn(("OLD", "local_only"), statuses)

    def test_diff_supported_models_uses_inventory(self):
        upstream = {
            "architectures": [
                {"architecture": "Qwen3ForCausalLM", "models": ["Qwen3-32B"]},
                {"architecture": "BrandNewArch", "models": ["X"]},
            ]
        }
        inventory = {
            "inventory": [
                {"architecture": "Qwen3ForCausalLM", "type": "llm", "models": ["Qwen3-32B"]},
                {"architecture": "OldArch", "type": "llm", "models": []},
            ]
        }
        entries = diff_supported_models(upstream, inventory)
        statuses = {(e.field, e.status) for e in entries}
        self.assertIn(("BrandNewArch", "upstream_only"), statuses)
        self.assertIn(("OldArch", "local_only"), statuses)


class TestFetcherFallbacks(unittest.TestCase):
    def test_fixture_dir_short_circuits_http(self, *_):
        result = fetch_url(
            "vllm-cli-serve-reference",
            "https://example.invalid/should-not-be-hit",
            cache_dir=Path("/tmp/wings-test-cache"),
            use_cache=False,
            fixture_dir=FIXTURES,
        )
        self.assertTrue(result.from_cache)
        self.assertIn("--model", result.body)
        self.assertIsNone(result.error)

    def test_missing_fixture_and_no_cache_records_error(self):
        result = fetch_url(
            "vllm-cli-serve-reference",
            "https://example.invalid/no-such-url",
            cache_dir=Path("/tmp/wings-test-cache-empty"),
            use_cache=True,
            fixture_dir=Path("/nonexistent"),
            opener=lambda url: (_ for _ in ()).throw(OSError("simulated offline")),
        )
        self.assertIsNotNone(result.error)
        self.assertEqual(result.body, "")


class TestEndToEnd(unittest.TestCase):
    def test_run_sync_writes_reports_under_output_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = SyncConfig(
                repo_root=ROOT,
                config_root=CONFIG_ROOT,
                output_dir=tmp_path / "reports",
                cache_dir=tmp_path / "_raw",
                fixture_dir=FIXTURES,
                use_cache=False,
                engines=["vllm", "vllm_ascend"],
            )
            aggregate = run_sync(cfg)
            self.assertGreaterEqual(aggregate["summary"]["reports"], 3)
            self.assertEqual(aggregate["summary"]["fetch_errors"], 0)

            vllm_report = tmp_path / "reports" / "vllm" / "vllm-cli-serve-reference.json"
            self.assertTrue(vllm_report.is_file())
            payload = json.loads(vllm_report.read_text(encoding="utf-8"))
            self.assertEqual(payload["engine"], "vllm")
            self.assertEqual(payload["parse"]["kind"], "cli_params")
            # fixture has `new_flag_2026` not in local manifest → at least one upstream_only
            statuses = {entry["status"] for entry in payload["diff"]}
            self.assertIn("upstream_only", statuses)

            ascend_models = tmp_path / "reports" / "vllm_ascend" / "vllm-ascend-model-support.json"
            self.assertTrue(ascend_models.is_file())
            payload = json.loads(ascend_models.read_text(encoding="utf-8"))
            statuses = {entry["status"] for entry in payload["diff"]}
            # Real upstream model support table contains models not yet in inventory.
            self.assertIn("upstream_only", statuses)


if __name__ == "__main__":
    unittest.main()
