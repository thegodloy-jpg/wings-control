# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6.1 Phase D model_deploy_config compatibility carrier tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.model_deploy_compat_loader import load_migrated_model_deploy_config  # noqa: E402
from core.config_loader import _load_and_validate_models_dict  # noqa: E402


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestV36ModelDeployCompatLoader(unittest.TestCase):
    def _sample_model_deploy_config(self):
        return {
            "llm": {
                "default": {
                    "vllm": {"trust_remote_code": True, "max_model_len": 4096},
                    "sglang": {"trust_remote_code": True, "context_length": 4096},
                },
                "Qwen3ForCausalLM": {
                    "default": {
                        "vllm": {
                            "trust_remote_code": True,
                            "max_model_len": 8192,
                            "tool_call_parser": "hermes",
                        }
                    }
                },
            }
        }

    def test_load_migrated_model_deploy_config_from_phase_d_carrier(self):
        loaded, ref = load_migrated_model_deploy_config({"device": "nvidia"})

        self.assertIsNotNone(loaded)
        self.assertIn("llm", loaded)
        self.assertEqual(ref, "phase-d-carriers")

    def test_config_loader_prefers_migrated_carrier_when_present(self):
        migrated = self._sample_model_deploy_config()
        with patch(
            "core.config_loader.load_migrated_model_deploy_config",
            return_value=(migrated, "recipes/model_deploy_compat/nvidia.yaml"),
        ), patch("core.config_loader._load_default_config") as legacy_default:
            models_dict, fallback = _load_and_validate_models_dict({"device": "nvidia"}, "llm", "vllm")

        legacy_default.assert_not_called()
        self.assertIsNone(fallback)
        self.assertEqual(models_dict["Qwen3ForCausalLM"]["default"]["vllm"]["max_model_len"], 8192)

    def test_migration_tool_writes_compat_carrier(self):
        tool = _load_tool("migrate_model_deploy_config")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            defaults_dir = root / "defaults"
            defaults_dir.mkdir(parents=True)
            source = defaults_dir / "nvidia_default.json"
            source.write_text(
                json.dumps({"model_deploy_config": self._sample_model_deploy_config()}),
                encoding="utf-8",
            )

            stats = tool.migrate_device(root, "nvidia")
            target = root / "recipes" / "model_deploy_compat" / "nvidia.yaml"
            written = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(stats["device"], "nvidia")
        self.assertTrue(target.name, "nvidia.yaml")
        self.assertEqual(written["schema_version"], "model_deploy_compat.v1")
        self.assertIn("model_deploy_config", written)


if __name__ == "__main__":
    unittest.main()
