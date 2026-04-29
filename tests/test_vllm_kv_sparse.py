# -*- coding: utf-8 -*-
"""vLLM KV Sparse 启动参数构建逻辑单测。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：同时支持包路径导入和模块内历史顶层导入
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(ROOT))

from wings_control.engines import vllm_adapter  # noqa: E402


class _FakeModelInfo:
    def __init__(self, architecture: str):
        self.model_architecture = architecture


class TestVllmKvSparse(unittest.TestCase):
    def test_indexcache_arch_returns_hf_overrides_without_fp8_mutation(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine_config": {"kv_cache_dtype": "auto"},
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            extra_args = vllm_adapter._build_kv_sparse_cmd(params, "vllm")

        self.assertEqual(extra_args, " --hf-overrides '{\"index_topk_freq\": 4}'")
        self.assertEqual(params["engine_config"], {"kv_cache_dtype": "auto"})

    def test_non_indexcache_arch_injects_fp8_kv_cache_config(self):
        params = {
            "model_name": "qwen",
            "model_path": "/models/qwen",
            "model_type": "llm",
            "engine_config": {"kv_cache_dtype": "auto"},
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen2ForCausalLM"),
        ):
            extra_args = vllm_adapter._build_kv_sparse_cmd(params, "vllm")

        self.assertEqual(extra_args, "")
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")
        self.assertTrue(params["engine_config"]["calculate_kv_scales"])

    def test_non_vllm_engine_skips_sparse_logic_without_model_detection(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine_config": {"kv_cache_dtype": "auto"},
        }

        with patch.object(vllm_adapter, "ModelIdentifier") as mock_identifier:
            extra_args = vllm_adapter._build_kv_sparse_cmd(params, "vllm_ascend")

        self.assertEqual(extra_args, "")
        self.assertEqual(params["engine_config"], {"kv_cache_dtype": "auto"})
        mock_identifier.assert_not_called()

    def test_speculative_and_indexcache_sparse_are_appended_together(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine": "vllm",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "engine_config": {"model": "/models/deepseek-v32"},
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)

        self.assertIn(
            "--speculative-config '{\"method\": \"deepseek_mtp\", "
            "\"num_speculative_tokens\": 3}'",
            script,
        )
        self.assertIn("--hf-overrides '{\"index_topk_freq\": 4}'", script)
        self.assertLess(script.index("--speculative-config"), script.index("--hf-overrides"))

    def test_speculative_and_fp8_sparse_are_rendered_together(self):
        params = {
            "model_name": "qwen",
            "model_path": "/models/qwen",
            "model_type": "llm",
            "engine": "vllm",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "engine_config": {"model": "/models/qwen", "kv_cache_dtype": "auto"},
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("Qwen2ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)

        self.assertIn("--kv-cache-dtype fp8", script)
        self.assertIn("--calculate-kv-scales", script)
        self.assertIn(
            "--speculative-config '{\"method\" : \"suffix\", "
            "\"num_speculative_tokens\": 5, "
            "\"suffix_decoding_max_cached_requests\": 1000}'",
            script,
        )
        self.assertNotIn("--hf-overrides", script)

    def test_speculative_without_sparse_does_not_enable_sparse_args(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine": "vllm",
            "enable_speculative_decode": True,
            "enable_sparse": False,
            "engine_config": {"model": "/models/deepseek-v32", "kv_cache_dtype": "auto"},
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)

        self.assertIn("--speculative-config", script)
        self.assertNotIn("--hf-overrides", script)
        self.assertNotIn("--kv-cache-dtype fp8", script)
        self.assertNotIn("--calculate-kv-scales", script)

    def test_sparse_control_flag_is_not_rendered_as_native_vllm_arg(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine": "vllm",
            "enable_sparse": True,
            "engine_config": {
                "model": "/models/deepseek-v32",
                "enable_sparse": True,
            },
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)

        self.assertIn("--hf-overrides '{\"index_topk_freq\": 4}'", script)
        self.assertNotIn("--enable-sparse", script)

    def test_sparse_speculative_and_kv_offload_args_are_all_preserved(self):
        params = {
            "model_name": "deepseek-v32",
            "model_path": "/models/deepseek-v32",
            "model_type": "llm",
            "engine": "vllm",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "engine_config": {
                "model": "/models/deepseek-v32",
                "kv_transfer_config": {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                },
            },
        }

        with patch.object(
            vllm_adapter,
            "ModelIdentifier",
            return_value=_FakeModelInfo("DeepseekV32ForCausalLM"),
        ):
            script = vllm_adapter.build_start_script(params)

        self.assertIn("--kv-transfer-config", script)
        self.assertIn("LMCacheConnectorV1", script)
        self.assertIn(
            "--speculative-config '{\"method\": \"deepseek_mtp\", "
            "\"num_speculative_tokens\": 3}'",
            script,
        )
        self.assertIn("--hf-overrides '{\"index_topk_freq\": 4}'", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)