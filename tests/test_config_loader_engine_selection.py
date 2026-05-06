# -*- coding: utf-8 -*-
"""config_loader 引擎选择逻辑单测。"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import (  # noqa: E402
    _apply_us8_long_ctx_strategy,
    _detect_mtp_moe_features,
    _set_deepseek_v31_ascend_quant_params,
    _set_soft_fp8,
    _set_mindie_common_params,
    _select_ascend_engine,
    _select_nvidia_engine,
    _validate_user_engine,
)
from utils.model_utils import is_deepseek_series_fp8, is_deepseek_series_modelslim_quant  # noqa: E402


class _FakeModelInfo:
    def __init__(self, architecture="FakeCausalLM", model_type="generate", supported=True,
                 model_name="fake", model_path=""):
        self.model_architecture = architecture
        self._model_type = model_type
        self._supported = supported
        self.model_name = model_name
        self.model_path = model_path

    def identify_model_type(self):
        return self._model_type

    def is_wings_supported(self):
        return self._supported


class TestConfigLoaderEngineSelection(unittest.TestCase):
    def test_lmcache_no_longer_forces_nvidia_to_vllm(self):
        model_info = _FakeModelInfo(supported=True)
        with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False):
            engine = _select_nvidia_engine("full", model_info)
        self.assertEqual(engine, "sglang")

    def test_lmcache_no_longer_forces_ascend_to_vllm_ascend(self):
        model_info = _FakeModelInfo(supported=True)
        with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False):
            engine = _select_ascend_engine("Ascend910B", model_info)
        self.assertEqual(engine, "mindie")

    def test_validate_user_engine_keeps_mindie_with_lmcache(self):
        model_info = _FakeModelInfo(supported=True)
        with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False):
            engine = _validate_user_engine("mindie", "Ascend910B", "full", model_info)
        self.assertEqual(engine, "mindie")

    def test_mindie_moe_requires_enable_ep_moe(self):
        params = {"enable_ep_moe": False}

        _detect_mtp_moe_features({"model_name": "deepseek-r1-671b"}, params)

        self.assertEqual(params["isMOE"], False)

    def test_mindie_moe_enabled_by_enable_ep_moe(self):
        params = {"enable_ep_moe": True}

        _detect_mtp_moe_features({"model_name": "Qwen3-32B"}, params)

        self.assertEqual(params["isMOE"], True)

    def test_mindie_implicit_gpu_memory_default_does_not_set_npu_fraction(self):
        params = {}
        engine_cmd_parameter = {"gpu_memory_utilization": 0.9}

        with patch.object(sys, "argv", ["wings-launcher-v4"]):
            with patch.dict(os.environ, {}, clear=True):
                _set_mindie_common_params(params, engine_cmd_parameter)

        self.assertNotIn("npu_memory_fraction", params)

    def test_mindie_explicit_gpu_memory_can_set_npu_fraction(self):
        params = {}
        engine_cmd_parameter = {"gpu_memory_utilization": 0.95}

        with patch.object(sys, "argv", ["wings-launcher-v4", "--gpu-memory-utilization", "0.95"]):
            with patch.dict(os.environ, {}, clear=True):
                _set_mindie_common_params(params, engine_cmd_parameter)

        self.assertEqual(params["npu_memory_fraction"], 0.95)

    def test_mindie_deepseek_long_context_2x8_triggers_cpsp_defaults(self):
        params = {}
        ctx = {
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 8,
        }
        engine_cmd_parameter = {"input_length": 131072, "output_length": 1024}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")
        env = {"MINDIE_LONG_CONTEXT_THRESHOLD": "8192"}

        with patch.dict(os.environ, env, clear=True):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertEqual(params["dp"], 1)
        self.assertEqual(params["sp"], 8)
        self.assertEqual(params["cp"], 2)
        self.assertEqual(params["tp"], 8)
        self.assertEqual(params["maxSeqLen"], 132096)
        self.assertEqual(params["maxInputTokenLen"], 132096)
        self.assertEqual(params["maxPrefillTokens"], 132096)

    def test_mindie_deepseek_long_context_1x16_triggers_cpsp_defaults(self):
        params = {}
        ctx = {
            "distributed": False,
            "nnodes": 1,
            "device_count": 16,
        }
        engine_cmd_parameter = {"input_length": 8192, "output_length": 1}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")

        with patch.dict(os.environ, {"MINDIE_LONG_CONTEXT_THRESHOLD": "8192"}, clear=True):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertEqual(params["dp"], 1)
        self.assertEqual(params["sp"], 8)
        self.assertEqual(params["cp"], 2)
        self.assertEqual(params["tp"], 8)
        self.assertEqual(params["maxSeqLen"], 8193)
        self.assertEqual(params["maxInputTokenLen"], 8193)
        self.assertEqual(params["maxPrefillTokens"], 8193)

    def test_mindie_deepseek_long_context_2x16_does_not_auto_trigger_cpsp(self):
        params = {}
        ctx = {
            "distributed": True,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "device_count": 16,
        }
        engine_cmd_parameter = {"input_length": 8192, "output_length": 1}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")

        with patch.dict(os.environ, {"MINDIE_LONG_CONTEXT_THRESHOLD": "8192"}, clear=True):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertNotIn("sp", params)
        self.assertNotIn("cp", params)
        self.assertNotIn("tp", params)
        self.assertNotIn("dp", params)

    def test_mindie_deepseek_short_context_does_not_trigger_cpsp(self):
        params = {}
        ctx = {"distributed": True}
        engine_cmd_parameter = {"input_length": 4096, "output_length": 4096}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")

        with patch.dict(os.environ, {"MINDIE_LONG_CONTEXT_THRESHOLD": "8192"}, clear=False):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertNotIn("sp", params)
        self.assertNotIn("cp", params)

    def test_deepseek_v31_w8a8_uses_official_quant_path_not_soft_fp8(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}),
                encoding="utf-8",
            )
            (model_dir / "quant_model_description.json").write_text("{}", encoding="utf-8")
            model_info = _FakeModelInfo(
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                model_path=str(model_dir),
            )
            params = {"device_count": 8}

            handled = _set_deepseek_v31_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertEqual(params["quantization"], "ascend")
        self.assertNotIn("enforce_eager", params)
        self.assertEqual(params["tensor_parallel_size"], 4)
        self.assertEqual(params["data_parallel_size"], 2)

    def test_deepseek_v31_w8a8_preserves_explicit_upper_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}),
                encoding="utf-8",
            )
            (model_dir / "quant_model_description.json").write_text("{}", encoding="utf-8")
            model_info = _FakeModelInfo(
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                model_path=str(model_dir),
            )
            params = {
                "device_count": 8,
                "enforce_eager": True,
                "tensor_parallel_size": 8,
            }

            with patch.object(sys, "argv", ["prog", "--enforce-eager", "--tensor-parallel-size", "8"]):
                handled = _set_deepseek_v31_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertIs(params["enforce_eager"], True)
        self.assertEqual(params["tensor_parallel_size"], 8)
        self.assertEqual(params["data_parallel_size"], 2)

    def test_deepseek_v31_does_not_enter_soft_fp8_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}),
                encoding="utf-8",
            )
            (model_dir / "quant_model_description.json").write_text(
                json.dumps({"quant_type": "w8a8"}),
                encoding="utf-8",
            )
            model_info = _FakeModelInfo(
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                model_path=str(model_dir),
            )
            params = {"device_count": 8}

            self.assertTrue(is_deepseek_series_modelslim_quant(str(model_dir)))
            self.assertFalse(is_deepseek_series_fp8(str(model_dir)))
            _set_soft_fp8(params, {"device": "ascend"}, model_info)

        self.assertEqual(params, {"device_count": 8})

    def test_generic_deepseek_fp8_still_forces_enforce_eager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}),
                encoding="utf-8",
            )
            (model_dir / "quant_model_description.json").write_text(
                json.dumps({"quant_type": "fp8"}),
                encoding="utf-8",
            )
            model_info = _FakeModelInfo(
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-R1-w8a8",
                model_path=str(model_dir),
            )
            params = {"device_count": 8}

            _set_soft_fp8(params, {"device": "ascend"}, model_info)

        self.assertEqual(params["quantization"], "ascend")
        self.assertIs(params["enforce_eager"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
