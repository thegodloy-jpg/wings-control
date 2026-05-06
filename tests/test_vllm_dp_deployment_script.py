# -*- coding: utf-8 -*-
"""vLLM Ascend dp_deployment 启动脚本单测。"""
# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import build_start_script  # noqa: E402


class _FakeDeepSeekModelIdentifier:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "DeepseekV3ForCausalLM"
        self.model_quantize = ""


def _base_params(node_rank=0, enable_speculative_decode=False):
    engine_config = {
        "trust_remote_code": True,
        "max_model_len": 131072,
        "host": "10.254.124.178",
        "port": 17000,
        "served_model_name": "DeepSeek-V3.1-w8a8",
        "model": "/usr/local/serving/models/",
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "quantization": "ascend",
        "gpu_memory_utilization": 0.95,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,
        "block_size": 128,
        "max_num_seqs": 256,
        "seed": 42,
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "no_enable_prefix_caching": True,
    }
    return {
        "engine": "vllm_ascend",
        "distributed": True,
        "nnodes": 2,
        "node_rank": node_rank,
        "head_node_addr": "10.254.124.178",
        "master_ip": "10.254.124.178",
        "node_ips": "10.254.124.178,10.254.13.111",
        "distributed_executor_backend": "dp_deployment",
        "rpc_port": 27071,
        "nixl_port": 27070,
        "device_count": 8,
        "model_name": "DeepSeek-V3.1-w8a8",
        "model_path": "/usr/local/serving/models/",
        "model_type": "",
        "enable_speculative_decode": enable_speculative_decode,
        "engine_config": engine_config,
    }


class TestVllmDpDeploymentScript(unittest.TestCase):
    def test_deepseek_dp_deployment_rank0_uses_vllm_serve_not_ray(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=0))

        self.assertIn("exec vllm serve /usr/local/serving/models/", script)
        self.assertIn("--data-parallel-size 4", script)
        self.assertIn("--data-parallel-size-local 2", script)
        self.assertIn("--data-parallel-rank 0", script)
        self.assertIn("export HCCL_WHITELIST_DISABLE=1", script)
        self.assertIn("export HCCL_CONNECT_TIMEOUT=1800", script)
        self.assertIn("export HCCL_EXEC_TIMEOUT=7200", script)
        self.assertIn("${ASCEND_CUSTOM_OPP_PATH:-}", script)
        self.assertIn("${LD_LIBRARY_PATH:-}", script)
        self.assertNotIn("${ASCEND_CUSTOM_OPP_PATH}", script)
        self.assertNotIn("${LD_LIBRARY_PATH}", script)
        self.assertNotIn("ray start --head", script)
        self.assertNotIn("--distributed-executor-backend ray", script)
        self.assertNotIn("--speculative-config", script)

    def test_deepseek_dp_deployment_rank1_is_headless(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=1))

        self.assertIn("--headless", script)
        self.assertIn("--data-parallel-start-rank 2", script)
        self.assertIn("--data-parallel-size-local 2", script)
        self.assertNotIn("ray start --address", script)
        self.assertNotIn("--host 10.254.124.178", script)
        self.assertNotIn("--port 17000", script)

    def test_deepseek_dp_deployment_speculative_switch_appends_mtp(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=0, enable_speculative_decode=True))

        self.assertIn("export VLLM_EARS_TOLERANCE=0.5", script)
        self.assertIn(
            "--speculative-config '{\"method\": \"deepseek_mtp\", "
            "\"num_speculative_tokens\": 3}'",
            script,
        )
        self.assertLess(script.index("--speculative-config"), script.index("--data-parallel-address"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
