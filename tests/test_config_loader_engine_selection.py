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

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from core.config_loader import (  # noqa: E402
    _apply_us8_long_ctx_strategy,
    _detect_mtp_moe_features,
    load_and_merge_configs,
    _set_deepseek_v31_ascend_quant_params,
    _set_deepseek_v3_family_ascend_quant_params,
    _set_sequence_length,
    _set_soft_fp8,
    _guard_pd_hybrid_kv_cache,
    _set_mindie_common_params,
    _select_ascend_engine,
    _select_nvidia_engine,
    _validate_user_engine,
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

    def test_glm5_vllm_ascend_cli_flags_and_env_render_to_start_script(self):
        from core.start_args_compat import parse_launch_args  # noqa: E402
        from engines.vllm_adapter import build_start_script  # noqa: E402

        with tempfile.TemporaryDirectory() as model_dir:
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
                "--async-scheduling",
                "--additional-config", '{"fuse_muls_add": true}',
                "--speculative-config", '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}',
                "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
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
        self.assertIn("--additional-config '{\"fuse_muls_add\":true}'", exec_line)
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

    def test_deepseek_v31_w8a8_uses_official_quant_path_not_soft_fp8(self):
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

            handled = _set_deepseek_v31_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertEqual(params["quantization"], "ascend")
        self.assertEqual(params["dtype"], "bfloat16")
        self.assertNotIn("enforce_eager", params)
        self.assertEqual(params["tensor_parallel_size"], 4)
        self.assertEqual(params["data_parallel_size"], 2)

    def test_deepseek_v31_w8a8_preserves_explicit_upper_params(self):
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
            params = {
                "device_count": 8,
                "enforce_eager": True,
                "tensor_parallel_size": 8,
                "dtype": "float16",
            }

            with patch.object(sys, "argv", [
                "prog", "--enforce-eager", "--tensor-parallel-size", "8", "--dtype", "float16"
            ]):
                handled = _set_deepseek_v31_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertIs(params["enforce_eager"], True)
        self.assertEqual(params["tensor_parallel_size"], 8)
        self.assertEqual(params["data_parallel_size"], 2)
        self.assertEqual(params["dtype"], "float16")

    def test_deepseek_v31_does_not_enter_soft_fp8_branch(self):
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
                architecture="DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                model_path=str(model_dir),
            )
            params = {"device_count": 8}

            self.assertTrue(is_deepseek_series_modelslim_quant(str(model_dir)))
            self.assertFalse(is_deepseek_series_fp8(str(model_dir)))
            _set_soft_fp8(params, {"device": "ascend"}, model_info)

        self.assertEqual(params, {"device_count": 8})

    def test_deepseek_v3_family_w8a8_reuses_bfloat16_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["DeepseekV32ForCausalLM"]}),
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
            params = {"device_count": 8, "dtype": "auto", "enforce_eager": True}

            handled = _set_deepseek_v3_family_ascend_quant_params(params, {"device": "ascend"}, model_info)

        self.assertTrue(handled)
        self.assertEqual(params["dtype"], "bfloat16")
        self.assertNotIn("enforce_eager", params)
        self.assertEqual(params["quantization"], "ascend")
        self.assertEqual(params["tensor_parallel_size"], 4)
        self.assertEqual(params["data_parallel_size"], 2)

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
            "compilation_config": "",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
