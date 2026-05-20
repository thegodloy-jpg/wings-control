# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6.1 Phase D runtime loader tests."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.phase_d_loader import PhaseDRuntimeContext, build_phase_d_resolution  # noqa: E402
from core.config_loader import _maybe_apply_phase_d_layers, load_and_merge_configs  # noqa: E402
from core.provenance import build_resolution_trace  # noqa: E402


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


def _known_args(**overrides):
    base = {
        "engine": "vllm_ascend",
        "engine_version": "0.18.0",
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
        "config_file": "",
        "engine_config": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestV36PhaseDLoader(unittest.TestCase):
    def test_resolution_loads_manifest_deviation_and_env_policy(self):
        ctx = PhaseDRuntimeContext(engine="vllm_ascend", version="0.18.0", chip_id="910b-32")
        with patch.dict(os.environ, {}, clear=True):
            result = build_phase_d_resolution(ctx, {"gpu_memory_utilization": 0.9})
        self.assertTrue(result["summary"]["manifest_found"])
        self.assertGreaterEqual(result["summary"]["deviation_count"], 1)
        # vllm_ascend env_policies now covers the full forced-env set from
        # set_vllm_ascend_env.sh plus runtime-rendered env policy entries;
        # USE_KUNLUN_ATB is 910c-only so 910b-32 sees one fewer entry.
        self.assertGreaterEqual(result["summary"]["env_policy_count"], 2)
        self.assertEqual(result["proposed_params"]["gpu_memory_utilization"], 0.92)
        self.assertEqual(result["proposed_env"]["VLLM_USE_MODELSCOPE"], "False")

    def test_shadow_mode_does_not_mutate_engine_config(self):
        engine_config = {"gpu_memory_utilization": 0.9}
        with patch.dict(os.environ, {"WINGS_PHASE_D_MODE": "shadow"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]):
            result = _maybe_apply_phase_d_layers(
                engine_config,
                {"chip_id": "910b-32"},
                {"engine": "vllm_ascend", "engine_version": "0.18.0"},
            )
        self.assertIs(result, engine_config)
        self.assertEqual(engine_config["gpu_memory_utilization"], 0.9)

    def test_primary_mode_applies_deviation_unless_explicit(self):
        with patch.dict(os.environ, {"WINGS_PHASE_D_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]):
            engine_config = {"gpu_memory_utilization": 0.9}
            _maybe_apply_phase_d_layers(
                engine_config,
                {"chip_id": "910b-32"},
                {"engine": "vllm_ascend", "engine_version": "0.18.0"},
            )
        self.assertEqual(engine_config["gpu_memory_utilization"], 0.92)

        with patch.dict(os.environ, {"WINGS_PHASE_D_MODE": "primary"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4", "--gpu-memory-utilization", "0.8"]):
            engine_config = {"gpu_memory_utilization": 0.8}
            _maybe_apply_phase_d_layers(
                engine_config,
                {"chip_id": "910b-32"},
                {"engine": "vllm_ascend", "engine_version": "0.18.0"},
            )
        self.assertEqual(engine_config["gpu_memory_utilization"], 0.8)

    def test_resolution_trace_marks_phase_d_sources(self):
        ctx = PhaseDRuntimeContext(engine="vllm_ascend", version="0.18.0", chip_id="910b-32")
        with patch.dict(os.environ, {}, clear=True):
            phase_d = build_phase_d_resolution(ctx, {"gpu_memory_utilization": 0.9})
        trace = build_resolution_trace(
            {"gpu_memory_utilization": 0.92},
            phase_d_resolution=phase_d,
            legacy_ref="legacy.json",
        )
        self.assertEqual(trace["cli.gpu_memory_utilization"]["source_layer"], "deviation")
        self.assertIn("vllm_ascend.yaml#gpu_memory_utilization", trace["cli.gpu_memory_utilization"]["source_ref"])
        self.assertEqual(trace["env.VLLM_USE_MODELSCOPE"]["source_layer"], "env_policy")
        self.assertIn("vllm_ascend.yaml#VLLM_USE_MODELSCOPE", trace["env.VLLM_USE_MODELSCOPE"]["source_ref"])

    def test_load_and_merge_configs_phase_d_primary_trace(self):
        with patch.dict(os.environ, {"WINGS_PHASE_D_MODE": "primary", "WINGS_RECIPE_MODE": "shadow"}, clear=True), \
                patch.object(sys, "argv", ["wings-launcher-v4"]), \
                patch("core.config_loader.ModelIdentifier", return_value=_FakeModelInfo()), \
                patch("core.config_loader._check_vram_requirements", return_value=None), \
                patch("core.config_loader._get_model_specific_config", return_value={"gpu_memory_utilization": 0.9}):
            merged = load_and_merge_configs(
                {"device": "ascend", "chip_id": "910b-32", "details": []},
                _known_args(),
            )

        self.assertEqual(merged["engine_config"]["gpu_memory_utilization"], 0.92)
        trace = merged["_resolution_trace"]
        self.assertEqual(trace["cli.gpu_memory_utilization"]["source_layer"], "deviation")
        self.assertEqual(trace["env.VLLM_USE_MODELSCOPE"]["source_layer"], "env_policy")


if __name__ == "__main__":
    unittest.main()
