# -*- coding: utf-8 -*-
"""Kimi-K2.5 vLLM-Ascend 适配单测。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

import engines.vllm_adapter as vllm_adapter  # noqa: E402
from utils.model_utils import ModelIdentifier  # noqa: E402


class _FakeKimiModel:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "KimiK25ForConditionalGeneration"
        self.model_quantize = ""


class TestKimiK25AscendDefaults(unittest.TestCase):
    def test_ascend_default_contains_kimi_k25_vllm_fields(self):
        defaults_path = ROOT / "wings_control" / "config" / "defaults" / "ascend_default.json"
        defaults = json.loads(defaults_path.read_text(encoding="utf-8-sig"))

        cfg = defaults["model_deploy_config"]["llm"]["KimiK25ForConditionalGeneration"]["default"]["vllm_ascend"]

        self.assertEqual(cfg["quantization"], "ascend")
        self.assertEqual(cfg["tool_call_parser"], "kimi_k2")
        self.assertEqual(cfg["reasoning_parser"], "kimi_k2")
        self.assertEqual(cfg["mm_encoder_tp_mode"], "data")
        self.assertTrue(cfg["enable_expert_parallel"])
        self.assertTrue(cfg["async_scheduling"])
        self.assertNotIn("no_enable_prefix_caching", cfg)
        self.assertEqual(cfg["speculative_config"]["method"], "eagle3")
        self.assertEqual(cfg["speculative_config"]["num_speculative_tokens"], 3)
        self.assertEqual(cfg["compilation_config"]["cudagraph_mode"], "FULL_DECODE_ONLY")

    def test_model_identifier_marks_kimi_k25_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["KimiK25ForConditionalGeneration"]}),
                encoding="utf-8",
            )

            info = ModelIdentifier("Kimi-K2.5", str(model_dir), "auto")

            self.assertEqual(info.model_architecture, "KimiK25ForConditionalGeneration")
            self.assertEqual(info.identify_model_type(), "llm")
            self.assertTrue(info.is_wings_supported())


class TestKimiK25AscendStartScript(unittest.TestCase):
    def _build_params(self):
        return {
            "engine": "vllm_ascend",
            "distributed": False,
            "model_name": "Kimi-K2.5",
            "model_path": "/usr/local/serving/models",
            "model_type": "auto",
            "enable_speculative_decode": False,
            "engine_config": {
                "model": "/usr/local/serving/models",
                "host": "0.0.0.0",
                "port": 18000,
                "quantization": "ascend",
                "served_model_name": "kimi-k25",
                "allowed_local_media_path": "/",
                "trust_remote_code": True,
                "no_enable_prefix_caching": True,
                "seed": 1024,
                "tensor_parallel_size": 16,
                "data_parallel_size": 1,
                "enable_expert_parallel": True,
                "tool_call_parser": "kimi_k2",
                "reasoning_parser": "kimi_k2",
                "enable_auto_tool_choice": True,
                "async_scheduling": True,
                "max_num_seqs": 16,
                "max_model_len": 44000,
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.9,
                "compilation_config": {
                    "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 128, 256],
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                },
                "speculative_config": {
                    "method": "eagle3",
                    "model": "/models",
                    "num_speculative_tokens": 3,
                },
                "mm_encoder_tp_mode": "data",
            },
        }

    def test_start_script_contains_kimi_env_and_cli_fields(self):
        with patch.object(vllm_adapter, "ModelIdentifier", _FakeKimiModel):
            script = vllm_adapter.build_start_script(self._build_params())

        self.assertIn("export HCCL_OP_EXPANSION_MODE=AIV", script)
        self.assertIn("export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True", script)
        self.assertIn("export TASK_QUEUE_ENABLE=1", script)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1", script)
        self.assertIn("export VLLM_ASCEND_BALANCE_SCHEDULING=1", script)
        self.assertIn("export VLLM_ENGINE_READY_TIMEOUT_S=3600", script)
        self.assertIn("WINGS_ASCEND_PERF_TUNING", script)
        self.assertIn("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true", script)
        self.assertIn("sysctl -w vm.swappiness=0 || true", script)
        self.assertIn("sysctl -w kernel.numa_balancing=0 || true", script)
        self.assertIn("sysctl -w kernel.sched_migration_cost_ns=50000 || true", script)

        self.assertIn("--quantization ascend", script)
        self.assertIn("--tool-call-parser kimi_k2", script)
        self.assertIn("--reasoning-parser kimi_k2", script)
        self.assertIn("--enable-auto-tool-choice", script)
        self.assertIn("--async-scheduling", script)
        self.assertIn("--mm-encoder-tp-mode data", script)
        self.assertIn('"num_speculative_tokens":3', script)
        self.assertEqual(script.count("--speculative-config"), 2)  # echoed command + exec command


if __name__ == "__main__":
    unittest.main()
