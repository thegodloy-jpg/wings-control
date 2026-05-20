# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6.1 Phase D mapping loader tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.mapping_loader import get_engine_parameter_mapping  # noqa: E402
from core.config_loader import _translate_user_config_for_engine  # noqa: E402


class TestV36MappingLoader(unittest.TestCase):
    def test_vllm_mapping_replaces_legacy_json_shape(self):
        mapping = get_engine_parameter_mapping("vllm")
        self.assertEqual(mapping["model_name"], "served_model_name")
        self.assertEqual(mapping["model_path"], "model")
        self.assertEqual(mapping["gpu_memory_utilization"], "gpu_memory_utilization")
        self.assertEqual(mapping["input_length"], "")

    def test_vllm_ascend_uses_canonical_engine_mapping(self):
        mapping = get_engine_parameter_mapping("vllm-ascend")
        self.assertEqual(mapping["model_path"], "model")
        self.assertEqual(mapping["enable_prefix_caching"], "enable_prefix_caching")

    def test_sglang_and_mindie_mappings(self):
        sglang = get_engine_parameter_mapping("sglang")
        mindie = get_engine_parameter_mapping("mindie")
        self.assertEqual(sglang["gpu_memory_utilization"], "mem_fraction_static")
        self.assertEqual(sglang["enable_prefix_caching"], "disable_chunked_prefix_cache")
        self.assertEqual(mindie["model_name"], "modelName")
        self.assertEqual(mindie["gpu_memory_utilization"], "npu_memory_fraction")
        self.assertEqual(mindie["dtype"], "")

    def test_translate_user_config_uses_phase_d_mapping(self):
        translated = _translate_user_config_for_engine(
            {"model_path": "/models/m", "gpu_memory_utilization": 0.8, "max_model_len": 4096},
            "sglang",
        )
        self.assertEqual(translated["model_path"], "/models/m")
        self.assertEqual(translated["mem_fraction_static"], 0.8)
        self.assertEqual(translated["context_length"], 4096)


if __name__ == "__main__":
    unittest.main()
