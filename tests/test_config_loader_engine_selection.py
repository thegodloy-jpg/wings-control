# -*- coding: utf-8 -*-
"""config_loader 引擎选择逻辑单测。"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import (  # noqa: E402
    _apply_us8_long_ctx_strategy,
    _detect_mtp_moe_features,
    _select_ascend_engine,
    _select_nvidia_engine,
    _validate_user_engine,
)


class _FakeModelInfo:
    def __init__(self, architecture="FakeCausalLM", model_type="generate", supported=True):
        self.model_architecture = architecture
        self._model_type = model_type
        self._supported = supported

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

    def test_mindie_deepseek_long_context_triggers_cpsp_defaults(self):
        params = {}
        ctx = {"distributed": True}
        engine_cmd_parameter = {"input_length": 8192, "output_length": 1}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")
        env = {
            "MINDIE_LONG_CONTEXT_THRESHOLD": "8192",
            "MINDIE_DS_DP": "1",
            "MINDIE_DS_SP": "8",
            "MINDIE_DS_CP": "2",
            "MINDIE_DS_TP": "2",
        }

        with patch.dict(os.environ, env, clear=False):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertEqual(params["dp"], 1)
        self.assertEqual(params["sp"], 8)
        self.assertEqual(params["cp"], 2)
        self.assertEqual(params["tp"], 2)

    def test_mindie_deepseek_short_context_does_not_trigger_cpsp(self):
        params = {}
        ctx = {"distributed": True}
        engine_cmd_parameter = {"input_length": 4096, "output_length": 4096}
        model_info = _FakeModelInfo(architecture="DeepseekV3ForCausalLM")

        with patch.dict(os.environ, {"MINDIE_LONG_CONTEXT_THRESHOLD": "8192"}, clear=False):
            _apply_us8_long_ctx_strategy(params, ctx, engine_cmd_parameter, model_info)

        self.assertNotIn("sp", params)
        self.assertNotIn("cp", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
