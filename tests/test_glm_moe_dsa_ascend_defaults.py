# -*- coding: utf-8 -*-
"""GLM-5 / GlmMoeDsaForCausalLM vLLM-Ascend 默认参数单测。"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

import engines.vllm_adapter as vllm_adapter  # noqa: E402


class _FakeGlmMoeDsaModel:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "GlmMoeDsaForCausalLM"
        self.model_quantize = ""


class TestGlmMoeDsaAscendDefaults(unittest.TestCase):
    def _glm_cfg(self):
        defaults_path = ROOT / "wings_control" / "config" / "defaults" / "ascend_default.json"
        defaults = json.loads(defaults_path.read_text(encoding="utf-8-sig"))
        return defaults["model_deploy_config"]["llm"]["GlmMoeDsaForCausalLM"]["default"]["vllm_ascend"]

    def test_ascend_default_matches_glm5_vllm_018_fields(self):
        cfg = self._glm_cfg()

        self.assertEqual(cfg["quantization"], "ascend")
        self.assertTrue(cfg["enable_expert_parallel"])
        self.assertEqual(cfg["seed"], 1024)
        self.assertEqual(cfg["max_num_seqs"], 8)
        self.assertEqual(cfg["max_model_len"], 4096)
        self.assertEqual(cfg["max_num_batched_tokens"], 4096)
        self.assertEqual(cfg["gpu_memory_utilization"], 0.95)
        self.assertTrue(cfg["enable_chunked_prefill"])
        self.assertTrue(cfg["enable_prefix_caching"])
        self.assertTrue(cfg["async_scheduling"])
        self.assertEqual(cfg["additional_config"]["fuse_muls_add"], True)
        self.assertEqual(cfg["additional_config"]["multistream_overlap_shared_expert"], True)
        self.assertEqual(cfg["additional_config"]["ascend_compilation_config"]["enable_npugraph_ex"], True)
        self.assertEqual(cfg["compilation_config"], {"cudagraph_mode": "FULL_DECODE_ONLY"})
        self.assertEqual(cfg["speculative_config"], {"num_speculative_tokens": 3, "method": "deepseek_mtp"})
        self.assertEqual(cfg["tool_call_parser"], "glm47")
        self.assertEqual(cfg["reasoning_parser"], "glm45")

    def test_start_script_with_defaults_contains_glm5_fields(self):
        cfg = dict(self._glm_cfg())
        cfg.update({
            "model": "/usr/local/serving/models/",
            "host": "0.0.0.0",
            "port": 18000,
            "served_model_name": "glm-5",
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
        })
        params = {
            "engine": "vllm_ascend",
            "distributed": False,
            "model_name": "GLM-5",
            "model_path": "/usr/local/serving/models/",
            "model_type": "auto",
            "enable_speculative_decode": False,
            "engine_config": cfg,
        }

        with patch.object(vllm_adapter, "ModelIdentifier", _FakeGlmMoeDsaModel):
            script = vllm_adapter.build_start_script(params)

        self.assertIn("export HCCL_OP_EXPANSION_MODE=AIV", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export HCCL_BUFFSIZE=200", script)
        self.assertIn("export VLLM_ASCEND_BALANCE_SCHEDULING=1", script)
        self.assertIn("--enable-prefix-caching", script)
        self.assertIn("--additional-config", script)
        self.assertIn('"fuse_muls_add":true', script)
        self.assertIn("--compilation-config", script)
        self.assertIn('"cudagraph_mode":"FULL_DECODE_ONLY"', script)
        self.assertIn("--speculative-config", script)
        self.assertIn('"method":"deepseek_mtp"', script)
        self.assertIn("--tool-call-parser glm47", script)
        self.assertIn("--reasoning-parser glm45", script)


if __name__ == "__main__":
    unittest.main()
