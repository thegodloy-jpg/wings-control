# -*- coding: utf-8 -*-
"""config_loader 引擎选择逻辑单测。"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 让测试可独立运行：同时支持包路径导入和模块内历史顶层导入
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(ROOT))

from wings_control.core.config_loader import (  # noqa: E402
    _apply_us8_long_ctx_strategy,
    _apply_engine_runtime_flags,
    _detect_explicit_cli_keys,
    _detect_mtp_moe_features,
    load_and_merge_configs,
    _resolve_engine_choice,
    _set_deepseek_v3_family_ascend_quant_params,
    _set_sequence_length,
    _set_soft_fp8,
    _guard_pd_hybrid_kv_cache,
    _set_mindie_common_params,
    _select_ascend_engine,
    _select_nvidia_engine,
    _validate_user_engine,
    _validate_embedding_rerank_params,
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

    def identify_model_architecture(self):
        return self.model_architecture

    def is_wings_supported(self):
        return self._supported


class TestConfigLoaderEngineSelection(unittest.TestCase):
    def test_vllm_native_env_keys_are_engine_scoped(self):
        with patch.object(sys, "argv", ["wings-launcher-v4"]):
            with patch.dict(os.environ, {
                "WINGS_ENGINE": "mindie",
                "GPU_MEMORY_UTILIZATION": "0.95",
                "ENFORCE_EAGER": "true",
                "TENSOR_PARALLEL_SIZE": "8",
            }, clear=True):
                keys = _detect_explicit_cli_keys()
        self.assertIn("gpu_memory_utilization", keys)
        self.assertNotIn("enforce_eager", keys)
        self.assertNotIn("tensor_parallel_size", keys)

    def test_vllm_native_env_keys_are_enabled_for_vllm_ascend(self):
        with patch.object(sys, "argv", ["wings-launcher-v4"]):
            with patch.dict(os.environ, {
                "WINGS_ENGINE": "vllm_ascend",
                "ENFORCE_EAGER": "true",
                "TENSOR_PARALLEL_SIZE": "8",
            }, clear=True):
                keys = _detect_explicit_cli_keys()
        self.assertIn("enforce_eager", keys)
        self.assertIn("tensor_parallel_size", keys)

    def test_vllm_advanced_flags_are_disabled_for_mindie(self):
        params = {
            "engine": "mindie",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "enable_rag_acc": False,
        }
        with patch.dict(os.environ, {}, clear=True):
            _apply_engine_runtime_flags(params)
            self.assertEqual(os.environ["SD_ENABLE"], "false")
            self.assertEqual(os.environ["SPARSE_ENABLE"], "false")
        self.assertFalse(params["enable_speculative_decode"])
        self.assertFalse(params["enable_sparse"])

    def test_vllm_advanced_flags_remain_enabled_for_vllm_ascend(self):
        params = {
            "engine": "vllm_ascend",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "enable_rag_acc": False,
        }
        with patch.dict(os.environ, {}, clear=True):
            _apply_engine_runtime_flags(params)
            self.assertEqual(os.environ["SD_ENABLE"], "true")
            self.assertEqual(os.environ["SPARSE_ENABLE"], "true")
        self.assertTrue(params["enable_speculative_decode"])
        self.assertTrue(params["enable_sparse"])

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

    def test_glm51_nvidia_forces_disable_kv_offload_even_with_upstream_config(self):
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "_name_or_path": "THUDM/GLM-5.1",
                    "torch_dtype": "bfloat16",
                }),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm",
                "--model-name", "GLM-5.1",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
            ]

            launch_args = parse_launch_args(argv).to_namespace()
            launch_args.engine_config = {
                "kv_transfer_config": {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
            }

            with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False), \
                    self.assertLogs("wings_control.core.config_loader", level="WARNING") as cm:
                merged = load_and_merge_configs(
                    {"device": "nvidia", "count": 1, "details": []},
                    launch_args,
                )

        self.assertNotIn("kv_transfer_config", merged["engine_config"])
        self.assertIn("Forced disabled for GLM-5.1 on NVIDIA/vLLM", "\n".join(cm.output))

    def test_glm5_nvidia_still_allows_lmcache_kv_offload(self):
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "_name_or_path": "THUDM/GLM-5",
                    "torch_dtype": "bfloat16",
                }),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm",
                "--model-name", "GLM-5",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
            ]

            launch_args = parse_launch_args(argv).to_namespace()
            with patch.dict(os.environ, {"LMCACHE_OFFLOAD": "true"}, clear=False):
                merged = load_and_merge_configs(
                    {"device": "nvidia", "count": 1, "details": []},
                    launch_args,
                )

        self.assertIn("kv_transfer_config", merged["engine_config"])
        self.assertIn("LMCacheConnectorV1", merged["engine_config"]["kv_transfer_config"])

    def test_nvidia_sparse_auto_selects_vllm(self):
        model_info = _FakeModelInfo(supported=True)

        engine = _resolve_engine_choice(
            "nvidia",
            "H100",
            "full",
            {"enable_sparse": True},
            model_info,
        )

        self.assertEqual(engine, "vllm")

    def test_nvidia_speculative_auto_selects_vllm(self):
        model_info = _FakeModelInfo(supported=True)

        engine = _resolve_engine_choice(
            "nvidia",
            "H100",
            "full",
            {"enable_speculative_decode": True},
            model_info,
        )

        self.assertEqual(engine, "vllm")

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

    def test_pd_removes_explicit_hybrid_kv_flag(self):
        params = {"no_disable_hybrid_kv_cache_manager": True}

        with patch.dict(os.environ, {"PD_ROLE": "P"}, clear=True):
            _guard_pd_hybrid_kv_cache(params)

        self.assertNotIn("no_disable_hybrid_kv_cache_manager", params)

    def test_non_pd_keeps_explicit_hybrid_kv_flag(self):
        params = {"no_disable_hybrid_kv_cache_manager": True}

        with patch.dict(os.environ, {}, clear=True):
            _guard_pd_hybrid_kv_cache(params)

        self.assertTrue(params["no_disable_hybrid_kv_cache_manager"])

    def test_glm5_vllm_ascend_engine_config_render_to_start_script(self):
        from core.start_args_compat import parse_launch_args  # noqa: E402
        from engines.vllm_adapter import build_start_script  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            config_file = Path(model_dir, "engine_config.json")
            config_file.write_text(
                json.dumps({
                    "async_scheduling": True,
                    "additional_config": {"fuse_muls_add": True},
                    "speculative_config": {"num_speculative_tokens": 3, "method": "deepseek_mtp"},
                    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
                }),
                encoding="utf-8",
            )
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "torch_dtype": "bfloat16",
                }),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm_ascend",
                "--model-name", "glm-5",
                "--model-path", model_dir,
                "--config-file", str(config_file),
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "8",
                "--enable-expert-parallel",
                "--seed", "1024",
                "--max-num-seqs", "8",
                "--max-num-batched-tokens", "4096",
                "--trust-remote-code",
                "--gpu-memory-utilization", "0.95",
                "--quantization", "ascend",
                "--enable-chunked-prefill",
                "--enable-prefix-caching",
            ]

            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {}, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "ascend", "count": 8, "details": []},
                        launch_args.to_namespace(),
                    )
                    script = build_start_script(merged)

        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--async-scheduling", exec_line)
        # GLM-5/5.1 on vllm_ascend A2（默认 platform）会深合并 additional_config 默认值：
        # 用户传入的 fuse_muls_add=true 被保留，multistream_overlap_shared_expert 与
        # ascend_compilation_config.enable_npugraph_ex 由 _apply_glm5_ascend_engine_defaults 补齐。
        self.assertIn("--additional-config '{\"fuse_muls_add\":true", exec_line)
        self.assertIn("\"multistream_overlap_shared_expert\":true", exec_line)
        self.assertIn("\"ascend_compilation_config\":{\"enable_npugraph_ex\":true}", exec_line)
        self.assertIn(
            "--speculative-config '{\"num_speculative_tokens\":3,\"method\":\"deepseek_mtp\"}'",
            exec_line,
        )
        self.assertIn("--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'", exec_line)
        self.assertEqual(exec_line.count("--speculative-config"), 1)

        for env_name in (
            "HCCL_OP_EXPANSION_MODE",
            "OMP_PROC_BIND",
            "OMP_NUM_THREADS",
            "HCCL_BUFFSIZE",
            "PYTORCH_NPU_ALLOC_CONF",
            "VLLM_ASCEND_BALANCE_SCHEDULING",
        ):
            self.assertIn(env_name, script)

    def _build_deepseek_v4_flash_script(self, platform="A2", extra_env=None, extra_argv=None, device_count=8):
        from core.start_args_compat import parse_launch_args  # noqa: E402
        from engines.vllm_adapter import build_start_script  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "torch_dtype": "bfloat16",
                }),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm_ascend",
                "--model-name", "DeepSeek-V4-Flash-w8a8-mtp",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", str(device_count),
                "--trust-remote-code",
            ]
            if extra_argv:
                argv.extend(extra_argv)

            env = {"WINGS_ASCEND_PLATFORM": platform}
            if extra_env:
                env.update(extra_env)
            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, env, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "ascend", "count": device_count, "details": []},
                        launch_args.to_namespace(),
                    )
                    return build_start_script(merged)

    def test_deepseek_v4_flash_a2_vllm_ascend_script(self):
        script = self._build_deepseek_v4_flash_script("A2")
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]

        self.assertTrue(exec_line.startswith("exec vllm serve "))
        self.assertIn("--max-model-len 65536", exec_line)
        self.assertIn("--max-num-batched-tokens 8192", exec_line)
        self.assertIn("--tensor-parallel-size 8", exec_line)
        self.assertIn("--data-parallel-size 1", exec_line)
        self.assertIn("--enable-expert-parallel", exec_line)
        self.assertIn("--quantization ascend", exec_line)
        self.assertIn("--block-size 128", exec_line)
        self.assertIn("--async-scheduling", exec_line)
        self.assertIn("--tokenizer-mode deepseek_v4", exec_line)
        self.assertIn("--tool-call-parser deepseek_v4", exec_line)
        self.assertIn("--enable-auto-tool-choice", exec_line)
        self.assertIn("--reasoning-parser deepseek_v4", exec_line)
        self.assertIn(
            "--speculative-config '{\"num_speculative_tokens\":1,\"method\":\"deepseek_mtp\"}'",
            exec_line,
        )
        self.assertIn("\"multistream_overlap_shared_expert\":true", exec_line)
        self.assertNotIn("ascend_compilation_config", exec_line)
        self.assertNotIn("multistream_dsa_preprocess", exec_line)
        self.assertEqual(exec_line.count("--speculative-config"), 1)
        self.assertEqual(exec_line.count("--additional-config"), 1)
        self.assertEqual(exec_line.count("--compilation-config"), 1)
        self.assertIn("--enable-prefix-caching", exec_line)
        self.assertIn("--safetensors-load-strategy prefetch", exec_line)
        self.assertNotIn("--use-vllm-serve", exec_line)
        self.assertNotIn("--ascend-platform", exec_line)
        self.assertIn("export OMP_NUM_THREADS=10", script)
        self.assertIn("export TRITON_ALL_BLOCKS_PARALLEL=1", script)
        self.assertNotIn("export USE_MULTI_GROUPS_KV_CACHE=1", script)
        self.assertNotIn("--kv-transfer-config", exec_line)
        # V4-Flash 不再默认强开 IndexCache：未带 --enable-sparse 时不应出现 --hf-overrides
        self.assertNotIn("--hf-overrides", exec_line)

    def test_deepseek_v4_flash_a3_vllm_ascend_script(self):
        script = self._build_deepseek_v4_flash_script("A3")
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]

        self.assertTrue(exec_line.startswith("exec vllm serve "))
        # V4-Flash 始终锁 TP=8（避免 MTP/sparse 小层被切到 0 维）
        self.assertIn("--tensor-parallel-size 8", exec_line)
        # 测试用 device_count=8，TP=8 → DP=1
        self.assertIn("--data-parallel-size 1", exec_line)
        self.assertIn("\"multistream_overlap_shared_expert\":false", exec_line)
        self.assertIn("\"multistream_dsa_preprocess\":false", exec_line)
        self.assertIn("\"ascend_compilation_config\"", exec_line)
        self.assertIn("\"enable_npugraph_ex\":true", exec_line)
        self.assertIn("\"enable_static_kernel\":false", exec_line)
        self.assertIn("--enable-prefix-caching", exec_line)
        self.assertIn("--safetensors-load-strategy prefetch", exec_line)
        self.assertIn("export OMP_NUM_THREADS=10", script)
        self.assertIn("export ASCEND_A3_ENABLE=1", script)
        self.assertIn("export USE_MULTI_GROUPS_KV_CACHE=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FUSED_MC2=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1", script)
        self.assertNotIn("export TRITON_ALL_BLOCKS_PARALLEL=1", script)
        self.assertNotIn("--kv-transfer-config", exec_line)
        # V4-Flash 不再默认强开 IndexCache：未带 --enable-sparse 时不应出现 --hf-overrides
        self.assertNotIn("--hf-overrides", exec_line)

    def test_deepseek_v4_flash_a3_16cards_tp_locked_to_8(self):
        """真实 A3 单机 16 卡 → TP 不再被拉满到 16，强制 TP=8；DP=16/8=2 把卡用满。"""
        script = self._build_deepseek_v4_flash_script("A3", device_count=16)
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 8", exec_line)
        self.assertNotIn("--tensor-parallel-size 16", exec_line)
        self.assertIn("--data-parallel-size 2", exec_line)

    def test_deepseek_v4_flash_a2_8cards_tp_locked_to_8(self):
        """A2 单机 8 卡 → TP=8、DP=1。"""
        script = self._build_deepseek_v4_flash_script("A2", device_count=8)
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 8", exec_line)
        self.assertIn("--data-parallel-size 1", exec_line)

    def test_deepseek_v4_flash_a3_distributed_two_nodes_dp_is_four(self):
        """A3 双机 × 16 卡 = 32 卡，TP=8 → DP=32/8=4。"""
        script = self._build_deepseek_v4_flash_script(
            "A3",
            device_count=16,
            extra_argv=[
                "--distributed", "--nnodes", "2",
                "--node-ips", "10.0.0.1,10.0.0.2",
                "--master-ip", "10.0.0.1",
            ],
            extra_env={"RANK_IP": "10.0.0.1"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 8", exec_line)
        self.assertIn("--data-parallel-size 4", exec_line)

    def test_deepseek_v4_flash_user_explicit_tp_respected(self):
        """用户显式 --tensor-parallel-size 16 → 完全尊重，不被强制覆盖到 8。"""
        script = self._build_deepseek_v4_flash_script(
            "A3",
            device_count=16,
            extra_env={"TENSOR_PARALLEL_SIZE": "16"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 16", exec_line)

    def test_deepseek_v4_flash_user_explicit_dp_respected(self):
        """用户显式 DP=4 → 完全尊重，不被自动 DP 推导覆盖。"""
        script = self._build_deepseek_v4_flash_script(
            "A3", extra_env={"DATA_PARALLEL_SIZE": "4"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--data-parallel-size 4", exec_line)

    def test_deepseek_v4_flash_a3_kv_offload_injects_cpu_offloading_connector(self):
        script = self._build_deepseek_v4_flash_script(
            "A3",
            extra_env={"LMCACHE_OFFLOAD": "true", "LMCACHE_MAX_LOCAL_CPU_SIZE": "100"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]

        self.assertIn("--kv-transfer-config", exec_line)
        self.assertIn("\"kv_connector\":\"CPUOffloadingConnector\"", exec_line)
        self.assertIn(
            "\"kv_connector_module_path\":\"vllm_ascend.distributed.kv_transfer"
            ".kv_pool.cpu_offload.cpu_offload_connector\"",
            exec_line,
        )
        self.assertIn("\"kv_role\":\"kv_both\"", exec_line)
        self.assertIn("\"swap_in_threshold\":1", exec_line)
        self.assertIn("\"cpu_swap_space_gb\":100", exec_line)
        # 与 MTP 共存：spec 不应降级
        self.assertIn(
            "--speculative-config '{\"num_speculative_tokens\":1,\"method\":\"deepseek_mtp\"}'",
            exec_line,
        )
        # LMCache env 与 YAML 路径不应再导出
        self.assertNotIn("export LMCACHE_OFFLOAD=", script)
        self.assertNotIn("export LMCACHE_CONFIG_FILE=", script)
        self.assertNotIn("export PYTHONHASHSEED=0", script)

    def test_deepseek_v4_flash_a2_kv_offload_uses_same_connector(self):
        script = self._build_deepseek_v4_flash_script(
            "A2",
            extra_env={"LMCACHE_OFFLOAD": "true", "LMCACHE_MAX_LOCAL_CPU_SIZE": "150"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]

        self.assertIn("\"kv_connector\":\"CPUOffloadingConnector\"", exec_line)
        self.assertIn("\"cpu_swap_space_gb\":150", exec_line)
        self.assertNotIn("export LMCACHE_OFFLOAD=", script)

    def test_deepseek_v4_flash_kv_offload_defaults_cpu_swap_to_200(self):
        script = self._build_deepseek_v4_flash_script(
            "A3", extra_env={"LMCACHE_OFFLOAD": "true"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]

        self.assertIn("\"cpu_swap_space_gb\":200", exec_line)

    def _build_deepseek_v4_pro_script(
        self,
        device_count=16,
        nnodes=2,
        node_rank=0,
        extra_env=None,
        extra_argv=None,
        model_name="DeepSeek-V4-Pro-w4a8-mtp",
    ):
        from core.start_args_compat import parse_launch_args  # noqa: E402
        from engines.vllm_adapter import build_start_script  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "torch_dtype": "bfloat16",
                }),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm_ascend",
                "--model-name", model_name,
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", str(device_count),
                "--trust-remote-code",
                "--distributed",
                "--nnodes", str(nnodes),
                "--node-rank", str(node_rank),
                "--node-ips", "10.0.0.1,10.0.0.2",
                "--master-ip", "10.0.0.1",
            ]
            if extra_argv:
                argv.extend(extra_argv)

            env = {"WINGS_ASCEND_PLATFORM": "A3", "RANK_IP": "10.0.0.1"}
            if extra_env:
                env.update(extra_env)
            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, env, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "ascend", "count": device_count, "details": []},
                        launch_args.to_namespace(),
                    )
                    return build_start_script(merged)

    def test_deepseek_v4_pro_a3_two_nodes_tp16_dp2_rank0(self):
        """V4-Pro A3 双机 16 卡：rank0 → TP=16、DP=2、dp_size_local=1、start_rank=0。"""
        script = self._build_deepseek_v4_pro_script(node_rank=0)
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertTrue(exec_line.startswith("exec vllm serve "))
        self.assertIn("--tensor-parallel-size 16", exec_line)
        self.assertIn("--data-parallel-size 2", exec_line)
        self.assertIn("--data-parallel-size-local 1", exec_line)
        # rank0 走 head 入口，没有 --headless / --data-parallel-start-rank
        self.assertNotIn("--headless", exec_line)
        self.assertNotIn("--data-parallel-start-rank", exec_line)
        self.assertIn("--max-model-len 135000", exec_line)
        self.assertIn("--max-num-batched-tokens 4096", exec_line)
        self.assertIn("--enable-expert-parallel", exec_line)
        self.assertIn("--quantization ascend", exec_line)
        self.assertIn("--block-size 128", exec_line)
        self.assertIn("--async-scheduling", exec_line)
        self.assertIn("--tokenizer-mode deepseek_v4", exec_line)
        self.assertIn("--tool-call-parser deepseek_v4", exec_line)
        self.assertIn("--enable-auto-tool-choice", exec_line)
        self.assertIn("--reasoning-parser deepseek_v4", exec_line)
        self.assertIn("--safetensors-load-strategy prefetch", exec_line)
        self.assertIn(
            "--speculative-config '{\"num_speculative_tokens\":1,\"method\":\"deepseek_mtp\"}'",
            exec_line,
        )
        self.assertIn("\"enable_cpu_binding\":\"true\"", exec_line)
        self.assertIn("\"enable_npugraph_ex\":true", exec_line)
        self.assertIn("\"enable_static_kernel\":false", exec_line)
        self.assertIn("export HCCL_BUFFSIZE=2048", script)
        self.assertIn("export ASCEND_A3_ENABLE=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FUSED_MC2=1", script)
        self.assertIn("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1", script)

    def test_deepseek_v4_pro_a3_two_nodes_rank1_headless_start_rank_1(self):
        """V4-Pro A3 双机 rank1 → --headless + --data-parallel-start-rank 1。"""
        script = self._build_deepseek_v4_pro_script(node_rank=1)
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 16", exec_line)
        self.assertIn("--data-parallel-size 2", exec_line)
        self.assertIn("--data-parallel-size-local 1", exec_line)
        self.assertIn("--headless", exec_line)
        self.assertIn("--data-parallel-start-rank 1", exec_line)

    def test_deepseek_v4_pro_user_explicit_tp_respected(self):
        """V4-Pro：用户显式 TP=8 应被尊重，不被强制覆盖到 16。"""
        script = self._build_deepseek_v4_pro_script(
            extra_env={"TENSOR_PARALLEL_SIZE": "8"},
        )
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        self.assertIn("--tensor-parallel-size 8", exec_line)

    def test_deepseek_v4_pro_flash_name_does_not_match_pro(self):
        """名称含 flash 严格视为 V4-Flash，不应进入 V4-Pro 分支。"""
        from engines.vllm_adapter import (
            _is_deepseek_v4_pro_params,
            _is_deepseek_v4_flash_params,
        )
        params = {"model_name": "DeepSeek-V4-Flash-w8a8-mtp"}
        self.assertFalse(_is_deepseek_v4_pro_params(params))
        self.assertTrue(_is_deepseek_v4_flash_params(params))

    def test_deepseek_v4_pro_single_node_does_not_apply_defaults(self):
        """V4-Pro 单机场景（不在适配范围）：不触发 V4-Pro 专属默认，HCCL 不应被改成 2048。"""
        from core.start_args_compat import parse_launch_args  # noqa: E402
        from engines.vllm_adapter import build_start_script  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}),
                encoding="utf-8",
            )
            argv = [
                "--engine", "vllm_ascend",
                "--model-name", "DeepSeek-V4-Pro-w4a8-mtp",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "16",
                "--trust-remote-code",
            ]
            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {"WINGS_ASCEND_PLATFORM": "A3"}, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "ascend", "count": 16, "details": []},
                        launch_args.to_namespace(),
                    )
                    script = build_start_script(merged)
        exec_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
        # 单机不在 V4-Pro 适配范围 → 不应注入 135000 / HCCL_BUFFSIZE=2048
        self.assertNotIn("--max-model-len 135000", exec_line)
        self.assertNotIn("export HCCL_BUFFSIZE=2048", script)

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

    def test_deepseek_v3_family_w8a8_uses_official_quant_path_not_soft_fp8(self):
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

            handled = _set_deepseek_v3_family_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertEqual(params["quantization"], "ascend")

    def test_deepseek_v3_family_modelslim_does_not_enter_soft_fp8_branch(self):
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
                architecture="DeepseekV32ForCausalLM",
                model_name="DeepSeek-V3.2-w8a8",
                model_path=str(model_dir),
            )
            params = {"device_count": 8}

            self.assertTrue(is_deepseek_series_modelslim_quant(str(model_dir)))
            self.assertFalse(is_deepseek_series_fp8(str(model_dir)))
            _set_soft_fp8(params, {"device": "ascend"}, model_info)

        self.assertEqual(params, {"device_count": 8})

    def test_distributed_raw_engine_config_is_preserved_as_explicit(self):
        known_args = SimpleNamespace()
        known_args.config_file = ""
        known_args._explicit_cli_keys = ["seed", "max_num_seqs"]
        known_args.engine_config = {
            "seed": 42,
            "max_num_seqs": 256,
            "max_model_len": 84096,
            "enable_prefix_caching": True,
        }
        for key, value in {
            "host": "0.0.0.0",
            "port": 17000,
            "model_name": "DeepSeek-V3.1-w8a8",
            "model_path": "/models/deepseek-v31",
            "engine": "vllm_ascend",
            "input_length": 4096,
            "output_length": 1024,
            "gpu_usage_mode": "full",
            "device_count": 8,
            "model_type": "auto",
            "save_path": "/tmp",
            "trust_remote_code": True,
            "dtype": "auto",
            "kv_cache_dtype": "auto",
            "quantization": "ascend",
            "quantization_param_path": "",
            "gpu_memory_utilization": 0.95,
            "enable_chunked_prefill": True,
            "block_size": 128,
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
            "distributed": True,
            "nnodes": 2,
            "node_rank": 1,
            "head_node_addr": "10.0.0.1",
            "distributed_executor_backend": "dp_deployment",
            "node_ips": "10.0.0.1,10.0.0.2",
            "nodes": "10.0.0.1,10.0.0.2",
            "master_ip": "10.0.0.1",
            "ray_head_ip": "10.0.0.1",
        }.items():
            setattr(known_args, key, value)

        with patch("core.config_loader._check_vram_requirements", return_value=None):
            with patch("core.config_loader.ModelIdentifier", lambda *args, **kwargs: _FakeModelInfo(
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                model_path="/models/deepseek-v31",
            )):
                merged = load_and_merge_configs({"device": "ascend", "device_type": "Ascend910B"}, known_args)

        engine_config = merged["engine_config"]
        self.assertEqual(engine_config["seed"], 42)
        self.assertEqual(engine_config["max_num_seqs"], 256)
        self.assertEqual(engine_config["max_model_len"], 84096)
        self.assertTrue(engine_config["enable_prefix_caching"])
        self.assertIn("seed", merged["_explicit_cli_keys"])
        self.assertIn("max_num_seqs", merged["_explicit_cli_keys"])
        self.assertNotIn("enable_prefix_caching", merged["_explicit_cli_keys"])

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

    def test_generic_deepseek_fp8_preserves_explicit_upper_overrides(self):
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
            params = {
                "device_count": 8,
                "enforce_eager": False,
                "tensor_parallel_size": 8,
            }

            with patch.object(sys, "argv", ["prog", "--enforce-eager", "false", "--tensor-parallel-size", "8"]):
                _set_soft_fp8(params, {"device": "ascend"}, model_info)

        self.assertIs(params["enforce_eager"], False)
        self.assertEqual(params["tensor_parallel_size"], 8)
        self.assertEqual(params["data_parallel_size"], 2)

    def test_sequence_length_does_not_override_without_explicit_input_or_output(self):
        params = {"max_model_len": 9999}

        with patch.object(sys, "argv", ["prog"]):
            _set_sequence_length(params, {"input_length": 4096, "output_length": 1024})

        self.assertEqual(params["max_model_len"], 9999)

    def test_sequence_length_applies_when_input_or_output_is_explicit(self):
        params = {"max_model_len": 9999}

        with patch.object(sys, "argv", ["prog", "--input-length", "4096"]):
            _set_sequence_length(params, {"input_length": 4096, "output_length": 1024})

        self.assertEqual(params["max_model_len"], 5120)


class TestModelIdentifierAutoDetect(unittest.TestCase):
    """ModelIdentifier.identify_model_type() - 空字符串/None 触发自动推断。"""

    def _make_identifier(self, model_name: str, model_type):
        from utils.model_utils import ModelIdentifier  # noqa: E402
        with patch("utils.model_utils.load_json_config", return_value={"architectures": []}):
            return ModelIdentifier(model_name, "/fake/path", model_type)

    def test_empty_string_triggers_auto_detect_rerank(self):
        """model_type='' 应与 'auto' 行为相同，能识别出 rerank 模型。"""
        mi = self._make_identifier("bge-reranker-v2-m3", "")
        self.assertEqual(mi.identify_model_type(), "rerank")

    def test_none_triggers_auto_detect_rerank(self):
        """model_type=None 应与 'auto' 行为相同，能识别出 rerank 模型。"""
        mi = self._make_identifier("bge-reranker-v2-m3", None)
        self.assertEqual(mi.identify_model_type(), "rerank")

    def test_auto_triggers_auto_detect_rerank(self):
        """model_type='auto' 原有行为保持不变。"""
        mi = self._make_identifier("bge-reranker-v2-m3", "auto")
        self.assertEqual(mi.identify_model_type(), "rerank")

    def test_empty_string_unknown_model_defaults_to_llm(self):
        """model_type='' 且 model_name 不在映射表时，默认返回 'llm'。"""
        mi = self._make_identifier("some-unknown-llm-model", "")
        self.assertEqual(mi.identify_model_type(), "llm")

    def test_explicit_embedding_bypasses_autodetect(self):
        """显式指定 model_type='embedding' 时直接返回，不做名称匹配。"""
        mi = self._make_identifier("bge-reranker-v2-m3", "embedding")
        self.assertEqual(mi.identify_model_type(), "embedding")

    def test_empty_string_triggers_auto_detect_embedding(self):
        """model_type='' 时 bge-large-en 能被识别为 embedding (如在映射表中)。"""
        # bge-large-en 若不在 embedding 映射表，仍应 fallback 为 llm
        mi = self._make_identifier("unknown-embedding-v1", "")
        self.assertIn(mi.identify_model_type(), ("embedding", "llm", "rerank"))


class TestValidateEmbeddingRerankFinalGuard(unittest.TestCase):
    """_validate_embedding_rerank_params 最终守卫 - 在所有合并之后执行。"""

    def setUp(self):
        from core.config_loader import _validate_embedding_rerank_params  # noqa: E402
        self._validate = _validate_embedding_rerank_params

    def test_embedding_removes_enable_prefix_caching(self):
        params = {"enable_prefix_caching": True, "max_model_len": 4096}
        self._validate(params, {"model_type": "embedding"})
        self.assertNotIn("enable_prefix_caching", params)

    def test_embedding_removes_enable_chunked_prefill(self):
        params = {"enable_chunked_prefill": True, "max_model_len": 4096}
        self._validate(params, {"model_type": "embedding"})
        self.assertNotIn("enable_chunked_prefill", params)

    def test_rerank_removes_both_incompatible_params(self):
        params = {"enable_prefix_caching": True, "enable_chunked_prefill": True}
        self._validate(params, {"model_type": "rerank"})
        self.assertNotIn("enable_prefix_caching", params)
        self.assertNotIn("enable_chunked_prefill", params)

    def test_llm_does_not_remove_any_params(self):
        params = {"enable_prefix_caching": True, "enable_chunked_prefill": True}
        self._validate(params, {"model_type": "llm"})
        self.assertIn("enable_prefix_caching", params)
        self.assertIn("enable_chunked_prefill", params)

    def test_empty_model_type_does_not_remove_params(self):
        """model_type='' 时不应触发清理（非 embedding/rerank）。"""
        params = {"enable_prefix_caching": True}
        self._validate(params, {"model_type": ""})
        self.assertIn("enable_prefix_caching", params)

    def test_embedding_removes_false_value_without_warning(self):
        """即使 enable_prefix_caching=False，也应将该 key 从 params 中移除。"""
        params = {"enable_prefix_caching": False}
        self._validate(params, {"model_type": "embedding"})
        self.assertNotIn("enable_prefix_caching", params)

    def test_load_and_merge_strips_prefix_caching_from_user_config(self):
        """回归：user_config 中的 enable_prefix_caching 不应出现在最终 engine_config 里。

        模拟 user_config 把 enable_prefix_caching 注入 engine_config，
        验证 load_and_merge_configs 末尾的最终守卫能正确清除它。
        """
        # 模拟 _merge_vllm_params 清理后被 user_config 重新注入的场景
        engine_config = {
            "max_model_len": 4096,
            "enable_prefix_caching": True,   # user_config / raw_engine_config 注入的
            "enable_chunked_prefill": True,
        }
        model_type = "embedding"

        _validate_embedding_rerank_params(engine_config, {"model_type": model_type})

        self.assertNotIn("enable_prefix_caching", engine_config)
        self.assertNotIn("enable_chunked_prefill", engine_config)
        # 其他参数保留
        self.assertEqual(engine_config["max_model_len"], 4096)


class TestSequenceLengthEmbeddingRerank(unittest.TestCase):
    """_set_sequence_length：embedding/rerank 只使用 input_length，不加 output_length。"""

    def test_llm_uses_input_plus_output(self):
        params = {}
        with patch.object(sys, "argv", ["prog", "--input-length", "4096", "--output-length", "1024"]):
            _set_sequence_length(params, {"input_length": 4096, "output_length": 1024}, model_type="llm")
        self.assertEqual(params["max_model_len"], 5120)

    def test_embedding_uses_only_input_length(self):
        params = {}
        with patch.object(sys, "argv", ["prog", "--input-length", "4096", "--output-length", "1024"]):
            _set_sequence_length(params, {"input_length": 4096, "output_length": 1024}, model_type="embedding")
        self.assertEqual(params["max_model_len"], 4096)

    def test_rerank_uses_only_input_length(self):
        params = {}
        with patch.object(sys, "argv", ["prog", "--input-length", "2048", "--output-length", "512"]):
            _set_sequence_length(params, {"input_length": 2048, "output_length": 512}, model_type="rerank")
        self.assertEqual(params["max_model_len"], 2048)

    def test_embedding_zero_output_length_result_equals_input_length(self):
        """output_length=0 时 embedding 结果应为 input_length。"""
        params = {}
        with patch.object(sys, "argv", ["prog", "--input-length", "8192"]):
            _set_sequence_length(params, {"input_length": 8192, "output_length": 0}, model_type="embedding")
        self.assertEqual(params["max_model_len"], 8192)

    def test_rerank_without_explicit_cli_does_not_set_max_model_len(self):
        """未显式传 --input-length / --output-length 时不应修改 params。"""
        params = {"max_model_len": 9999}
        with patch.object(sys, "argv", ["prog"]):
            _set_sequence_length(params, {"input_length": 4096, "output_length": 512}, model_type="rerank")
        self.assertEqual(params["max_model_len"], 9999)


class TestEmbeddingRerankE2E(unittest.TestCase):
    """embedding / rerank 模型在完整 load_and_merge_configs 流程中的端到端验证。

    覆盖两个 bug 修复：
    1. model_type='' (MODEL_TYPE 环境变量为空) 应能正确自动推断类型
    2. user_config / --engine-config 注入的 enable_prefix_caching 应在末尾被清除
    """

    def _make_rerank_model_dir(self, tmpdir: str, architecture="XLMRobertaForSequenceClassification"):
        model_dir = Path(tmpdir) / "bge-reranker-v2-m3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": [architecture], "torch_dtype": "float32"}),
            encoding="utf-8",
        )
        return str(model_dir)

    def test_rerank_auto_strips_prefix_caching_from_engine_config_cli_arg(self):
        """model_type='auto' + bge-reranker-v2-m3 命名匹配 → enable_prefix_caching 被剔除。"""
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_rerank_model_dir(tmpdir)
            argv = [
                "--engine", "vllm",
                "--model-name", "bge-reranker-v2-m3",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
                "--enable-prefix-caching",   # 用户手动传入，应被清除
                "--enable-chunked-prefill",  # 用户手动传入，应被清除
            ]

            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {}, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "nvidia", "count": 1, "details": []},
                        launch_args.to_namespace(),
                    )

        engine_cfg = merged.get("engine_config", {})
        self.assertNotIn("enable_prefix_caching", engine_cfg,
                         "rerank model must not have enable_prefix_caching in engine_config")
        self.assertNotIn("enable_chunked_prefill", engine_cfg,
                         "rerank model must not have enable_chunked_prefill in engine_config")

    def test_rerank_empty_model_type_env_strips_prefix_caching(self):
        """MODEL_TYPE='' 环境变量（空字符串）触发修复后的自动推断，仍能剔除不兼容参数。"""
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_rerank_model_dir(tmpdir)
            argv = [
                "--engine", "vllm",
                "--model-name", "bge-reranker-v2-m3",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
                "--enable-prefix-caching",
            ]

            # 模拟用户设置了 MODEL_TYPE=（空字符串）的环境变量
            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {"MODEL_TYPE": ""}, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "nvidia", "count": 1, "details": []},
                        launch_args.to_namespace(),
                    )

        engine_cfg = merged.get("engine_config", {})
        self.assertNotIn("enable_prefix_caching", engine_cfg,
                         "MODEL_TYPE='' should auto-detect rerank and strip enable_prefix_caching")

    def test_rerank_engine_config_injection_stripped_by_final_guard(self):
        """raw_engine_config（namespace.engine_config）注入 enable_prefix_caching，最终守卫应清除。

        这正是第二个 bug 的回归场景：raw_engine_config 在 _validate_embedding_rerank_params
        调用之后才被合并，修复后的最终守卫确保它依然被清除。
        """
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_rerank_model_dir(tmpdir)
            argv = [
                "--engine", "vllm",
                "--model-name", "bge-reranker-v2-m3",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
            ]

            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {}, clear=True):
                    launch_args = parse_launch_args(argv)
                    ns = launch_args.to_namespace()
                    # 模拟 raw_engine_config 路径（Worker 下发的已合并 engine_config）
                    ns.engine_config = {"enable_prefix_caching": True, "max_model_len": 4096}
                    merged = load_and_merge_configs(
                        {"device": "nvidia", "count": 1, "details": []},
                        ns,
                    )

        engine_cfg = merged.get("engine_config", {})
        self.assertNotIn("enable_prefix_caching", engine_cfg,
                         "raw_engine_config injected enable_prefix_caching must be stripped by final guard")

    def test_rerank_max_model_len_uses_only_input_length(self):
        """e2e：rerank 模型 max_model_len 只等于 input_length，不加 output_length。"""
        from core.start_args_compat import parse_launch_args  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._make_rerank_model_dir(tmpdir)
            argv = [
                "--engine", "vllm",
                "--model-name", "bge-reranker-v2-m3",
                "--model-path", model_dir,
                "--host", "0.0.0.0",
                "--port", "18000",
                "--device-count", "1",
                "--trust-remote-code",
                "--input-length", "4096",
                "--output-length", "1024",   # 应被忽略
            ]

            with patch.object(sys, "argv", ["wings-launcher-v4"] + argv):
                with patch.dict(os.environ, {}, clear=True):
                    launch_args = parse_launch_args(argv)
                    merged = load_and_merge_configs(
                        {"device": "nvidia", "count": 1, "details": []},
                        launch_args.to_namespace(),
                    )

        engine_cfg = merged.get("engine_config", {})
        self.assertEqual(engine_cfg.get("max_model_len"), 4096,
                         "rerank max_model_len must equal input_length only (4096), not input+output (5120)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
