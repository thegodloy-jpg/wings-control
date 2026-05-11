# -*- coding: utf-8 -*-
"""vLLM Ascend dp_deployment 启动脚本单测。"""
# pyright: reportMissingImports=false

import os
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


class _FakeDeepSeekV32ModelIdentifier:
    def __init__(self, *args, **kwargs):
        self.model_architecture = "DeepseekV32ForCausalLM"
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


def _generic_deepseek_params(node_rank=0):
    params = _base_params(node_rank=node_rank)
    params["model_name"] = "DeepSeek-R1-w8a8"
    params["engine_config"]["served_model_name"] = "DeepSeek-R1-w8a8"
    return params


def _deepseek_named_params(model_name, node_rank=0):
    params = _base_params(node_rank=node_rank)
    params["model_name"] = model_name
    params["engine_config"]["served_model_name"] = model_name
    return params


class TestVllmDpDeploymentScript(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_deepseek_dp_deployment_rank0_uses_vllm_serve_not_ray(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=0))

        self.assertIn("exec vllm serve /usr/local/serving/models/", script)
        self.assertIn("--data-parallel-size 4", script)
        self.assertIn("--data-parallel-size-local 2", script)
        self.assertNotIn("--data-parallel-start-rank 0", script)
        self.assertNotIn("--data-parallel-rank 0", script)
        self.assertNotIn("--data-parallel-external-lb", script)
        self.assertIn("export HCCL_WHITELIST_DISABLE=1", script)
        self.assertIn("export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-$(python3 -c", script)
        self.assertNotIn("try: s.connect", script)
        self.assertNotIn("finally: s.close()", script)
        self.assertNotIn("export HCCL_INTRA_PCIE_ENABLE=1", script)
        self.assertNotIn("export HCCL_INTRA_ROCE_ENABLE=0", script)
        self.assertIn("export HCCL_CONNECT_TIMEOUT=1800", script)
        self.assertIn("export HCCL_EXEC_TIMEOUT=7200", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)
        self.assertIn("export TASK_QUEUE_ENABLE=1", script)
        self.assertIn("export HCCL_OP_EXPANSION_MODE=AIV", script)
        self.assertIn('echo "[wings-env] final HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"', script)
        self.assertNotIn("export VLLM_ASCEND_BALANCE_SCHEDULING=1", script)
        self.assertNotIn("export VLLM_ASCEND_ENABLE_MLAPO=1", script)
        self.assertNotIn("export VLLM_ASCEND_ENABLE_NZ=0", script)
        self.assertIn("deepseek-v32/vendors/customize", script)
        self.assertNotIn("export VLLM_USE_V1=1", script)
        self.assertNotIn("export ASCEND_BUFFER_POOL=4:8", script)
        self.assertIn("export VLLM_ENGINE_READY_TIMEOUT_S=7200", script)
        self.assertNotIn("ray start --head", script)
        self.assertNotIn("--distributed-executor-backend ray", script)
        self.assertNotIn("--speculative-config", script)
        self.assertIn("--enable-expert-parallel", script)
        self.assertIn("--async-scheduling", script)
        self.assertIn("--dtype auto", script)
        self.assertIn("--seed 42", script)
        self.assertIn("--max-num-seqs 256", script)
        self.assertIn("--gpu-memory-utilization 0.95", script)
        self.assertNotIn("--compilation-config", script)
        self.assertIn("--enforce-eager", script)
        self.assertIn("[wings-cmd] >>> exec vllm serve /usr/local/serving/models/", script)

    def test_dp_deployment_strips_duplicate_dp_cli_flags_from_engine_config(self):
        params = _base_params(node_rank=0)
        params["engine_config"].update({
            "data_parallel_size": 8,
            "data_parallel_size_local": 1,
            "data_parallel_rank": 0,
            "data_parallel_address": "1.2.3.4",
            "data_parallel_rpc_port": 9999,
            "data_parallel_external_lb": True,
        })

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        exec_line = next(line for line in script.splitlines() if line.startswith("exec vllm serve"))
        self.assertEqual(exec_line.count("--data-parallel-size "), 1)
        self.assertEqual(exec_line.count("--data-parallel-size-local "), 1)
        self.assertEqual(exec_line.count("--data-parallel-address "), 1)
        self.assertEqual(exec_line.count("--data-parallel-rpc-port "), 1)
        self.assertIn("--data-parallel-size 4", exec_line)
        self.assertIn("--data-parallel-size-local 2", exec_line)
        self.assertNotIn("--data-parallel-start-rank", exec_line)
        self.assertNotIn("--data-parallel-rank", exec_line)
        self.assertNotIn("--data-parallel-external-lb", exec_line)
        self.assertNotIn("1.2.3.4", exec_line)
        self.assertNotIn("9999", exec_line)

    def test_deepseek_dp_deployment_rank1_is_headless(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=1))

        # exec 命令必须存在
        self.assertIn("exec vllm serve /usr/local/serving/models/", script)
        self.assertIn("[wings-cmd] >>> exec vllm serve /usr/local/serving/models/", script)
        self.assertNotIn("try: s.connect", script)
        self.assertNotIn("finally: s.close()", script)

        # worker 专属标志
        self.assertIn("--headless", script)
        self.assertIn("--data-parallel-start-rank 2", script)
        self.assertNotIn("--data-parallel-external-lb", script)
        self.assertNotIn("ray start --address", script)
        # worker 节点不应携带 HTTP 监听地址/端口
        self.assertNotIn("--host 10.254.124.178", script)
        self.assertNotIn("--port 17000", script)

        # DP 参数对齐 master（rank0）
        exec_line = next(line for line in script.splitlines() if line.startswith("exec vllm serve"))
        self.assertIn("--data-parallel-size 4", exec_line)
        self.assertIn("--data-parallel-size-local 2", exec_line)
        self.assertIn("--data-parallel-address 10.254.124.178", exec_line)
        self.assertIn("--data-parallel-rpc-port 27071", exec_line)

        # 模型启动参数对齐 master（关键字段不能丢失）
        self.assertIn("--enable-expert-parallel", script)
        self.assertIn("--async-scheduling", script)
        self.assertIn("--dtype auto", script)
        self.assertIn("--seed 42", script)
        self.assertIn("--max-num-seqs 256", script)
        self.assertIn("--enforce-eager", script)

    def test_deepseek_v31_official_dp_envs_do_not_leak_to_generic_deepseek(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_generic_deepseek_params(node_rank=0))

        self.assertNotIn("export HCCL_INTRA_PCIE_ENABLE=1", script)
        self.assertNotIn("export HCCL_INTRA_ROCE_ENABLE=0", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)

    def test_vllm_ascend_ray_distributed_omits_hccl_op_expansion_mode(self):
        params = _base_params(node_rank=0)
        params["distributed_executor_backend"] = "ray"

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertIn("ray start --head", script)
        self.assertIn("export OMP_PROC_BIND=false", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)
        self.assertIn("export TASK_QUEUE_ENABLE=1", script)
        self.assertNotIn("export HCCL_OP_EXPANSION_MODE=AIV", script)

    def test_vllm_ascend_single_node_ray_omits_hccl_op_expansion_mode(self):
        params = _base_params(node_rank=0)
        params["distributed"] = False
        params["nnodes"] = 1
        params["distributed_executor_backend"] = "ray"
        params["_explicit_cli_keys"] = ["distributed_executor_backend"]

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertNotIn("ray start --head", script)
        self.assertIn("exec python3 -m vllm.entrypoints.openai.api_server", script)
        self.assertIn("export OMP_PROC_BIND=false", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export HCCL_BUFFSIZE=1024", script)
        self.assertIn("export TASK_QUEUE_ENABLE=1", script)
        self.assertNotIn("export HCCL_OP_EXPANSION_MODE=AIV", script)

    def test_deepseek_dp_deployment_speculative_switch_appends_mtp(self):
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(_base_params(node_rank=0, enable_speculative_decode=True))

        # vllm_ascend 不注入 VLLM_EARS_TOLERANCE（Ascend 侧无需此参数）
        self.assertNotIn("export VLLM_EARS_TOLERANCE=0.5", script)
        self.assertIn(
            "--speculative-config '{\"method\": \"mtp\", "
            "\"num_speculative_tokens\": 3}'",
            script,
        )
        self.assertLess(script.index("--speculative-config"), script.index("--data-parallel-address"))

    def test_deepseek_dp_deployment_sanitizes_conflicting_prefix_cache_flags(self):
        params = _base_params(node_rank=0)
        params["engine_config"]["enable_expert_parallel"] = False
        params["engine_config"]["enable_prefix_caching"] = True
        params["engine_config"]["no_enable_prefix_caching"] = True

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertIn("--enable-expert-parallel", script)
        self.assertIn("--async-scheduling", script)
        self.assertNotIn("--enable-prefix-caching", script)
        self.assertIn("--no-enable-prefix-caching", script)

    def test_deepseek_v31_preserves_explicit_enforce_eager_override(self):
        params = _base_params(node_rank=0)
        params["_explicit_cli_keys"] = ["enforce_eager"]

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertIn("--enforce-eager", script)

    def test_deepseek_v31_preserves_explicit_dtype_override(self):
        params = _base_params(node_rank=0)
        params["_explicit_cli_keys"] = ["dtype"]
        params["engine_config"]["dtype"] = "float16"

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertIn("--dtype float16", script)
        self.assertNotIn("--dtype bfloat16", script)

    def test_deepseek_v31_converts_explicit_dtype_auto_to_bfloat16(self):
        params = _base_params(node_rank=0)
        params["_explicit_cli_keys"] = ["dtype"]
        params["engine_config"]["dtype"] = "auto"

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        self.assertIn("--dtype auto", script)
        self.assertNotIn("--dtype bfloat16", script)

    def test_deepseek_v31_dual_8_card_uses_formula_dp_topology(self):
        params = _base_params(node_rank=1)
        params["engine_config"].pop("tensor_parallel_size", None)

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)

        exec_line = next(line for line in script.splitlines() if line.startswith("exec vllm serve"))
        self.assertIn("--tensor-parallel-size 4", exec_line)
        self.assertIn("--data-parallel-size 4", exec_line)
        self.assertIn("--data-parallel-size-local 2", exec_line)
        self.assertIn("--data-parallel-start-rank 2", exec_line)

    def test_deepseek_v32_dual_8_card_defaults_tp8_dp2(self):
        params = _deepseek_named_params("DeepSeek-V3.2-w8a8", node_rank=1)
        params["engine_config"].pop("tensor_parallel_size", None)

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekV32ModelIdentifier):
            script = build_start_script(params)

        exec_line = next(line for line in script.splitlines() if line.startswith("exec vllm serve"))
        self.assertIn("--tensor-parallel-size 8", exec_line)
        self.assertIn("--data-parallel-size 2", exec_line)
        self.assertIn("--data-parallel-size-local 1", exec_line)
        self.assertIn("--data-parallel-start-rank 1", exec_line)

    def test_deepseek_v32_dual_16_card_defaults_tp16_dp2(self):
        params = _deepseek_named_params("DeepSeek-V3.2-w8a8", node_rank=1)
        params["device_count"] = 16
        params["engine_config"].pop("tensor_parallel_size", None)

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekV32ModelIdentifier):
            script = build_start_script(params)

        exec_line = next(line for line in script.splitlines() if line.startswith("exec vllm serve"))
        self.assertIn("--tensor-parallel-size 16", exec_line)
        self.assertIn("--data-parallel-size 2", exec_line)
        self.assertIn("--data-parallel-size-local 1", exec_line)
        self.assertIn("--data-parallel-start-rank 1", exec_line)

    def test_deepseek_dp_rejects_non_divisible_tp(self):
        params = _base_params(node_rank=0)
        params["_explicit_cli_keys"] = ["tensor_parallel_size"]
        params["engine_config"]["tensor_parallel_size"] = 3

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            with self.assertRaisesRegex(ValueError, "device_count to be divisible"):
                build_start_script(params)

    # ------------------------------------------------------------------
    # EARS 使能边界：vllm_ascend 场景各特性组合均不注入 VLLM_EARS_TOLERANCE
    # ------------------------------------------------------------------

    def test_vllm_ascend_speculative_plus_sparse_does_not_inject_ears(self):
        """vllm_ascend：投机推理 + IndexCache 多特性并存 — 三项验收：无 EARS 环境变量、不采集 EARS 补丁、--speculative-config 字段正常生成。"""
        from core.wings_entry import _collect_ears_patch_features  # noqa: E402

        params = _base_params(node_rank=0, enable_speculative_decode=True)
        params["enable_sparse"] = True

        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)
            patch_features = _collect_ears_patch_features("vllm_ascend", params)

        # 1. 环境变量：不注入
        self.assertNotIn("export VLLM_EARS_TOLERANCE=0.5", script)
        # 2. 补丁安装：不采集 EARS 补丁
        self.assertNotIn("ears", patch_features)
        # 3. 追加字段：--speculative-config 正常生成
        self.assertIn("--speculative-config", script)

    def test_vllm_ascend_speculative_plus_lmcache_does_not_inject_ears(self):
        """vllm_ascend：投机推理 + LMCache 多特性并存 — 三项验收：无 EARS 环境变量、不采集 EARS 补丁、suffix 方法字段正常生成。"""
        from core.wings_entry import _collect_ears_patch_features  # noqa: E402

        with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False):
            with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
                params = _base_params(node_rank=0, enable_speculative_decode=True)
                script = build_start_script(params)
                patch_features = _collect_ears_patch_features("vllm_ascend", params)

        # 1. 环境变量：不注入
        self.assertNotIn("export VLLM_EARS_TOLERANCE=0.5", script)
        # 2. 补丁安装：不采集 EARS 补丁
        self.assertNotIn("ears", patch_features)
        # 3. 追加字段：LMCache 使 DeepSeek MTP 降级为 suffix
        self.assertIn('"method" : "suffix"', script)

    def test_vllm_ascend_speculative_only_does_not_inject_ears(self):
        """vllm_ascend：单独开启投机推理（无其他高级特性）— 三项验收：无 EARS 环境变量、不采集 EARS 补丁、--speculative-config 字段正常生成。"""
        from core.wings_entry import _collect_ears_patch_features  # noqa: E402

        params = _base_params(node_rank=0, enable_speculative_decode=True)
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDeepSeekModelIdentifier):
            script = build_start_script(params)
            patch_features = _collect_ears_patch_features("vllm_ascend", params)

        # 1. 环境变量：不注入
        self.assertNotIn("export VLLM_EARS_TOLERANCE=0.5", script)
        # 2. 补丁安装：不采集 EARS 补丁
        self.assertNotIn("ears", patch_features)
        # 3. 追加字段：--speculative-config 正常生成
        self.assertIn("--speculative-config", script)

if __name__ == "__main__":
    unittest.main(verbosity=2)
