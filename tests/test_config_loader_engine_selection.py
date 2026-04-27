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
    _detect_mtp_moe_features,
    _select_ascend_engine,
    _select_nvidia_engine,
    _validate_user_engine,
)


class _FakeModelInfo:
    def __init__(self, architecture="FakeCausalLM", model_type="generate", supported=True, config=None):
        self.model_architecture = architecture
        self._model_type = model_type
        self._supported = supported
        self.config = config or {"architectures": [architecture]}

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

    def test_mindie_moe_detects_qwen3_moe_without_enable_ep(self):
        params = {"enable_ep_moe": False}
        model_info = _FakeModelInfo(
            architecture="Qwen3MoeForCausalLM",
            config={"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 128},
        )

        _detect_mtp_moe_features({"model_name": "Qwen3-30B-A3B"}, params, model_info)

        self.assertEqual(params["isMOE"], True)

    def test_mindie_moe_detects_deepseek_v3_without_enable_ep(self):
        params = {"enable_ep_moe": False}
        model_info = _FakeModelInfo(
            architecture="DeepseekV3ForCausalLM",
            config={"architectures": ["DeepseekV3ForCausalLM"], "n_routed_experts": 256},
        )

        _detect_mtp_moe_features({"model_name": "DeepSeek-V3.1"}, params, model_info)

        self.assertEqual(params["isMOE"], True)

    def test_mindie_moe_keeps_dense_model_false_without_enable_ep(self):
        params = {"enable_ep_moe": False}
        model_info = _FakeModelInfo(
            architecture="Qwen3ForCausalLM",
            config={"architectures": ["Qwen3ForCausalLM"]},
        )

        _detect_mtp_moe_features({"model_name": "Qwen3-32B"}, params, model_info)

        self.assertEqual(params["isMOE"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
