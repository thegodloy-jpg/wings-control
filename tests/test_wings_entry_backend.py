# -*- coding: utf-8 -*-
"""wings_entry 分布式后端透传单测。"""
# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.port_plan import PortPlan  # noqa: E402
from core.start_args_compat import LaunchArgs  # noqa: E402
from core.wings_entry import _prepare_merged_params  # noqa: E402


def _launch_args(**overrides):
    values = {
        "host": "0.0.0.0",
        "port": 18000,
        "model_name": "DeepSeek-V3.1-w8a8",
        "model_path": "/models/deepseek-v31",
        "engine": "vllm_ascend",
        "input_length": 4096,
        "output_length": 4096,
        "config_file": "",
        "gpu_usage_mode": "full",
        "device_count": 8,
        "model_type": "",
        "save_path": "",
        "trust_remote_code": True,
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "quantization": "",
        "quantization_param_path": "",
        "gpu_memory_utilization": 0.9,
        "enable_chunked_prefill": False,
        "block_size": 16,
        "max_num_seqs": 32,
        "seed": 0,
        "enable_expert_parallel": False,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": False,
        "enable_speculative_decode": False,
        "speculative_decode_model_path": "",
        "enable_rag_acc": False,
        "enable_auto_tool_choice": False,
        "enable_sparse": False,
        "compilation_config": "",
        "distributed": True,
        "nnodes": 2,
        "node_rank": 0,
        "head_node_addr": "10.0.0.1",
        "distributed_executor_backend": "ray",
        "node_ips": "10.0.0.1,10.0.0.2",
        "nodes": "10.0.0.1,10.0.0.2",
        "master_ip": "10.0.0.1",
        "ray_head_ip": "10.0.0.1",
    }
    values.update(overrides)
    return LaunchArgs(**values)


class TestWingsEntryBackend(unittest.TestCase):
    def test_load_merge_auto_selects_dp_for_ascend_deepseek(self):
        class _FakeDeepSeekModelInfo:
            model_architecture = "DeepseekV3ForCausalLM"
            model_name = "DeepSeek-V3.1-w8a8"
            model_path = "/models/deepseek-v31"

            def __init__(self, *args, **kwargs):
                pass

            def identify_model_architecture(self):
                return self.model_architecture

            def identify_model_type(self):
                return "llm"

        from core.config_loader import load_and_merge_configs

        launch_args = _launch_args().to_namespace()
        hardware = {"device": "ascend", "count": 8, "details": []}

        with patch("core.config_loader.ModelIdentifier", _FakeDeepSeekModelInfo):
            merged = load_and_merge_configs(hardware, launch_args)

        self.assertEqual(merged["distributed_executor_backend"], "dp_deployment")
        self.assertIn("rpc_port", merged)
        self.assertIn("nixl_port", merged)

    def test_preserves_auto_selected_dp_deployment_backend(self):
        merged_from_config_loader = {
            "engine": "vllm_ascend",
            "distributed_executor_backend": "dp_deployment",
            "engine_config": {},
        }
        port_plan = PortPlan(True, 17000, 18000, 19000, 19100)

        with patch("core.wings_entry.load_and_merge_configs", return_value=merged_from_config_loader):
            with patch("core.wings_entry._resolve_engine_service_host", return_value="10.0.0.1"):
                merged = _prepare_merged_params(_launch_args(), port_plan, {"device": "ascend"})

        self.assertEqual(merged["distributed_executor_backend"], "dp_deployment")

    def test_uses_launch_backend_when_merge_layer_has_no_backend(self):
        merged_from_config_loader = {
            "engine": "vllm_ascend",
            "engine_config": {},
        }
        port_plan = PortPlan(True, 17000, 18000, 19000, 19100)

        with patch("core.wings_entry.load_and_merge_configs", return_value=merged_from_config_loader):
            with patch("core.wings_entry._resolve_engine_service_host", return_value="10.0.0.1"):
                merged = _prepare_merged_params(_launch_args(distributed_executor_backend="ray"), port_plan, {"device": "ascend"})

        self.assertEqual(merged["distributed_executor_backend"], "ray")


if __name__ == "__main__":
    unittest.main(verbosity=2)
