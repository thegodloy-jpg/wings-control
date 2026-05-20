# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6 Phase B recipe-primary config loader tests."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import _get_model_specific_config, load_and_merge_configs  # noqa: E402


class _FakeModelInfo:
    model_architecture = "Qwen3ForCausalLM"
    model_name = "Qwen3-32B"
    model_path = "/models/Qwen3-32B"

    def identify_model_architecture(self):
        return self.model_architecture

    def identify_model_type(self):
        return "llm"

    def is_wings_supported(self):
        return True


class TestV36RecipePrimaryConfigLoader(unittest.TestCase):
    def _cmd(self, **overrides):
        base = {
            "engine": "vllm",
            "model_name": "Qwen3-32B",
            "model_path": "/models/Qwen3-32B",
            "model_type": "llm",
            "device_count": 1,
            "distributed": False,
            "nnodes": 1,
            "node_ips": "",
            "gpu_usage_mode": "full",
            "allow_experimental": False,
            "input_length": 4096,
            "output_length": 1024,
            "trust_remote_code": True,
            "max_num_seqs": 32,
            "enable_auto_tool_choice": False,
        }
        base.update(overrides)
        return base

    def test_primary_recipe_hit_skips_legacy_model_deploy_config(self):
        recipe_defaults = {"max_num_seqs": 64, "block_size": 128}
        resolution = {"status": "ok", "summary": {"selected_count": 1, "proposed_param_count": 2}}

        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]), \
                patch("core.config_loader.resolve_recipe_defaults", return_value=(recipe_defaults, resolution)), \
                patch("core.config_loader._load_and_validate_models_dict") as legacy_lookup:
            merged = _get_model_specific_config(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                self._cmd(),
                _FakeModelInfo(),
            )

        legacy_lookup.assert_not_called()
        self.assertEqual(merged["max_num_seqs"], 64)
        self.assertEqual(merged["block_size"], 128)

    def test_primary_recipe_hit_preserves_explicit_cli_override(self):
        recipe_defaults = {"max_num_seqs": 64, "block_size": 128}
        resolution = {"status": "ok", "summary": {"selected_count": 1, "proposed_param_count": 2}}

        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4", "--max-num-seqs", "16"]), \
                patch("core.config_loader.resolve_recipe_defaults", return_value=(recipe_defaults, resolution)), \
                patch("core.config_loader._load_and_validate_models_dict") as legacy_lookup:
            merged = _get_model_specific_config(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                self._cmd(max_num_seqs=16),
                _FakeModelInfo(),
            )

        legacy_lookup.assert_not_called()
        self.assertEqual(merged["max_num_seqs"], 16)
        self.assertEqual(merged["block_size"], 128)

    def test_primary_recipe_miss_falls_back_to_legacy_model_deploy_config(self):
        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]), \
                patch("core.config_loader.resolve_recipe_defaults", return_value=({}, {"status": "ok", "summary": {"selected_count": 0}})), \
                patch("core.config_loader._load_and_validate_models_dict", return_value=({}, {"legacy_default": True})) as legacy_lookup:
            merged = _get_model_specific_config(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                self._cmd(engine="unit"),
                _FakeModelInfo(),
            )

        legacy_lookup.assert_called_once()
        self.assertEqual(merged, {"legacy_default": True})

    def test_shadow_mode_keeps_legacy_model_deploy_config(self):
        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "shadow"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]), \
                patch("core.config_loader.resolve_recipe_defaults") as resolve_recipe, \
                patch("core.config_loader._load_and_validate_models_dict", return_value=({}, {"legacy_default": True})) as legacy_lookup:
            merged = _get_model_specific_config(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                self._cmd(engine="unit"),
                _FakeModelInfo(),
            )

        resolve_recipe.assert_not_called()
        legacy_lookup.assert_called_once()
        self.assertEqual(merged, {"legacy_default": True})

    def test_load_and_merge_configs_attaches_recipe_resolution_trace(self):
        recipe_defaults = {"max_num_seqs": 64}
        resolution = {
            "status": "ok",
            "summary": {"selected_count": 1, "proposed_param_count": 1},
            "selected": {
                "architecture": None,
                "model": {
                    "path": "wings_control/config/recipes/models/Qwen3-32B.yaml",
                    "defaults": recipe_defaults,
                },
            },
        }
        known_args = self._cmd()
        known_args.update({"config_file": "", "engine_config": None})

        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]), \
                patch("core.config_loader.ModelIdentifier", return_value=_FakeModelInfo()), \
                patch("core.config_loader._check_vram_requirements", return_value=None), \
                patch("core.config_loader.resolve_recipe_defaults", return_value=(recipe_defaults, resolution)):
            merged = load_and_merge_configs(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                SimpleNamespace(**known_args),
            )

        trace = merged["_resolution_trace"]
        self.assertIs(merged.merged, merged)
        self.assertIs(merged.as_dict(), merged)
        self.assertEqual(merged.resolution_trace, trace)
        self.assertEqual(merged.chip_info["chip_id"], "h20-141")
        self.assertEqual(len(merged.matched_recipes), 1)
        self.assertEqual(trace["cli.max_num_seqs"]["source_layer"], "model_recipe")
        self.assertNotIn("validation_tier", trace["cli.max_num_seqs"])
        self.assertNotIn("evidence_ref", trace["cli.max_num_seqs"])

    def test_resolution_trace_marks_explicit_cli_override(self):
        recipe_defaults = {"max_num_seqs": 64}
        resolution = {
            "status": "ok",
            "summary": {"selected_count": 1, "proposed_param_count": 1},
            "selected": {
                "architecture": None,
                "model": {
                    "path": "wings_control/config/recipes/models/Qwen3-32B.yaml",
                    "defaults": recipe_defaults,
                },
            },
        }
        known_args = self._cmd(max_num_seqs=16)
        known_args.update({"config_file": "", "engine_config": None})

        with patch.dict(os.environ, {"WINGS_RECIPE_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4", "--max-num-seqs", "16"]), \
                patch("core.config_loader.ModelIdentifier", return_value=_FakeModelInfo()), \
                patch("core.config_loader._check_vram_requirements", return_value=None), \
                patch("core.config_loader.resolve_recipe_defaults", return_value=(recipe_defaults, resolution)):
            merged = load_and_merge_configs(
                {"device": "nvidia", "chip_id": "h20-141", "details": []},
                SimpleNamespace(**known_args),
            )

        trace = merged["_resolution_trace"]
        self.assertEqual(merged["engine_config"]["max_num_seqs"], 16)
        self.assertEqual(trace["cli.max_num_seqs"]["source_layer"], "user_cli")
        self.assertEqual(trace["cli.max_num_seqs"]["source_ref"], "--max-num-seqs")


if __name__ == "__main__":
    unittest.main()
