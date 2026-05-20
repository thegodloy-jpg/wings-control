# -*- coding: utf-8 -*-
"""MiniMax-M2 vLLM-Ascend 适配单测。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

import engines.vllm_adapter as vllm_adapter  # noqa: E402
from core.model_deploy_compat_loader import load_migrated_model_deploy_config  # noqa: E402


class _FakeMiniMaxM2Model:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "MiniMaxM2ForCausalLM"
        self.model_quantize = ""


class TestMiniMaxM2AscendDefaults(unittest.TestCase):
    def test_ascend_default_contains_minimax_m2_vllm_fields(self):
        defaults, _ = load_migrated_model_deploy_config({"device": "ascend"})
        self.assertIsNotNone(defaults, "Phase D carrier assembly returned None")

        cfg = defaults["llm"]["MiniMaxM2ForCausalLM"]["default"]["vllm_ascend"]

        self.assertTrue(cfg["trust_remote_code"])
        self.assertEqual(cfg["gpu_memory_utilization"], 0.92)
        self.assertEqual(cfg["max_model_len"], 34816)
        self.assertEqual(cfg["max_num_seqs"], 64)
        self.assertEqual(cfg["tool_call_parser"], "minimax_m2")
        self.assertEqual(cfg["reasoning_parser"], "minimax_m2")
        self.assertEqual(cfg["compilation_config"], {"cudagraph_mode": "FULL_DECODE_ONLY"})
        self.assertEqual(cfg["additional_config"], {"enable_cpu_binding": True})

        distributed_cfg = defaults["llm"]["MiniMaxM2ForCausalLM"]["default"]["vllm_ascend_distributed"]
        self.assertEqual(distributed_cfg["compilation_config"], cfg["compilation_config"])
        self.assertEqual(distributed_cfg["additional_config"], cfg["additional_config"])


class TestMiniMaxM2AscendStartScript(unittest.TestCase):
    def _build_params(self):
        return {
            "engine": "vllm_ascend",
            "distributed": False,
            "model_name": "MiniMax-M2.5",
            "model_path": "/models/MiniMax-M2.5",
            "model_type": "auto",
            "enable_speculative_decode": False,
            "engine_config": {
                "model": "/models/MiniMax-M2.5",
                "host": "0.0.0.0",
                "port": 18000,
                "served_model_name": "MiniMax-M2.5",
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.92,
                "max_model_len": 34816,
                "max_num_seqs": 64,
                "tensor_parallel_size": 8,
                "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
                "additional_config": {"enable_cpu_binding": True},
                "tool_call_parser": "minimax_m2",
                "reasoning_parser": "minimax_m2_reasoning",
            },
        }

    def test_start_script_contains_minimax_env_and_cli_fields(self):
        with patch.object(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Model), patch.object(
            vllm_adapter, "is_deepseek_series_fp8", return_value=False
        ):
            script = vllm_adapter.build_start_script(self._build_params())

        # 通用 Ascend 初始化只保留在 set_vllm_ascend_env.sh 内联段中，MiniMax 专属段不再重复注入。
        export_lines = [line for line in script.splitlines() if line == "export HCCL_OP_EXPANSION_MODE=AIV"]
        self.assertEqual(len(export_lines), 1)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)
        self.assertIn("export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True", script)
        self.assertIn("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true", script)
        self.assertIn("sysctl -w vm.swappiness=0 || true", script)
        self.assertIn("sysctl -w kernel.numa_balancing=0 || true", script)
        self.assertIn("sysctl -w kernel.sched_migration_cost_ns=50000 || true", script)
        self.assertIn("libjemalloc.so.2", script)

        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export TASK_QUEUE_ENABLE=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FUSED_MC2=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1", script)
        self.assertIn("export VLLM_ASCEND_BALANCE_SCHEDULING=1", script)
        self.assertIn("export VLLM_USE_GRAPH=1", script)
        self.assertIn("export VLLM_USE_V1=1", script)
        self.assertIn("export VLLM_TORCH_COMPILE=0", script)

        self.assertIn("--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'", script)
        self.assertIn("--additional-config '{\"enable_cpu_binding\":true}'", script)
        self.assertIn("--tool-call-parser minimax_m2", script)
        self.assertIn("--reasoning-parser minimax_m2_reasoning", script)


if __name__ == "__main__":
    unittest.main()
