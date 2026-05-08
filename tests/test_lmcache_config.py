# -*- coding: utf-8 -*-
"""LMCache YAML and engine-side env export tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from engines import vllm_adapter  # noqa: E402


class TestLMCacheConfig(unittest.TestCase):
    def test_lmcache_yaml_uses_flat_offload_schema(self):
        env = {
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "64",
            "LMCACHE_LOCAL_DISK": "/cache/lmcache",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "128",
        }

        with patch.dict("os.environ", env, clear=True):
            config = vllm_adapter._build_lmcache_yaml_dict("vllm")

        self.assertEqual(config["chunk_size"], 256)
        self.assertIs(config["local_cpu"], True)
        self.assertEqual(config["max_local_cpu_size"], 64.0)
        self.assertEqual(config["local_disk"], "/cache/lmcache")
        self.assertEqual(config["max_local_disk_size"], 128.0)
        self.assertNotIsInstance(config["local_cpu"], dict)

    def test_lmcache_envs_are_exported_to_engine_script(self):
        env = {
            "LMCACHE_OFFLOAD": "true",
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "64",
            "LMCACHE_LOCAL_DISK": "/cache/lmcache",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "128",
        }

        with patch.dict("os.environ", env, clear=True), patch.object(
            vllm_adapter, "_write_lmcache_config_yaml", return_value="/shared-volume/lmcache_config.yaml"
        ):
            commands = vllm_adapter._build_cache_env_commands("vllm")

        self.assertIn("export PYTHONHASHSEED=0", commands)
        self.assertIn("export LMCACHE_OFFLOAD=true", commands)
        self.assertIn("export LMCACHE_LOCAL_CPU=true", commands)
        self.assertIn("export LMCACHE_MAX_LOCAL_CPU_SIZE=64", commands)
        self.assertIn("export LMCACHE_LOCAL_DISK=/cache/lmcache", commands)
        self.assertIn("export LMCACHE_MAX_LOCAL_DISK_SIZE=128", commands)
        self.assertIn("export LMCACHE_CONFIG_FILE=/shared-volume/lmcache_config.yaml", commands)

    def test_glm51_nvidia_skips_lmcache_env_exports(self):
        env = {
            "LMCACHE_OFFLOAD": "true",
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "64",
        }

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "_name_or_path": "THUDM/GLM-5.1",
                }),
                encoding="utf-8",
            )
            params = {
                "engine": "vllm",
                "model_name": "GLM-5.1",
                "model_path": model_dir,
                "model_type": "llm",
            }

            with patch.dict("os.environ", env, clear=True), \
                    self.assertLogs("engines.vllm_adapter", level="WARNING") as cm:
                commands = vllm_adapter._build_cache_env_commands("vllm", params)

        self.assertEqual(commands, [])
        self.assertIn("Forced disabled for GLM-5.1 on NVIDIA/vLLM", "\n".join(cm.output))

    def test_glm51_nvidia_removes_upstream_kv_transfer_config_from_script(self):
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "_name_or_path": "THUDM/GLM-5.1",
                }),
                encoding="utf-8",
            )
            params = {
                "engine": "vllm",
                "model_name": "GLM-5.1",
                "model_path": model_dir,
                "model_type": "llm",
                "engine_config": {
                    "model": model_dir,
                    "kv_transfer_config": {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
                },
            }

            with self.assertLogs("engines.vllm_adapter", level="WARNING") as cm:
                script = vllm_adapter.build_start_script(params)

        self.assertNotIn("--kv-transfer-config", script)
        self.assertNotIn("LMCacheConnectorV1", script)
        self.assertIn("Forced disabled for GLM-5.1 on NVIDIA/vLLM", "\n".join(cm.output))

    def test_cpu_size_implies_engine_side_local_cpu_enable(self):
        env = {
            "LMCACHE_OFFLOAD": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "64",
        }

        with patch.dict("os.environ", env, clear=True), patch.object(
            vllm_adapter, "_write_lmcache_config_yaml", return_value="/shared-volume/lmcache_config.yaml"
        ):
            commands = vllm_adapter._build_cache_env_commands("vllm")

        self.assertIn("export LMCACHE_LOCAL_CPU=true", commands)
        self.assertIn("export LMCACHE_MAX_LOCAL_CPU_SIZE=64", commands)

    def test_disk_only_offload_uses_flat_schema(self):
        env = {
            "LMCACHE_LOCAL_DISK": "/cache/lmcache",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "128",
        }

        with patch.dict("os.environ", env, clear=True):
            config = vllm_adapter._build_lmcache_yaml_dict("vllm")

        self.assertNotIn("local_cpu", config)
        self.assertEqual(config["local_disk"], "/cache/lmcache")
        self.assertEqual(config["max_local_disk_size"], 128.0)
        self.assertNotIsInstance(config["local_disk"], dict)

    def test_qat_and_cold_start_sections_are_preserved(self):
        env = {
            "LMCACHE_COLD_START": "true",
            "LMCACHE_QAT": "true",
            "LMCACHE_LOCAL_DISK": "/cache/lmcache",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "128",
            "LMCACHE_QAT_INSTANCE_NUM": "3",
            "LMCACHE_QAT_LOSS_LEVEL": "1",
            "LMCACHE_QAT_LOG_ENABLED": "1",
        }

        with patch.dict("os.environ", env, clear=True):
            config = vllm_adapter._build_lmcache_yaml_dict("vllm")

        self.assertEqual(config["local_disk"], "/cache/lmcache")
        self.assertEqual(config["max_local_disk_size"], 128.0)
        self.assertIn("pre_caching", config)
        self.assertEqual(config["pre_caching"]["hash_algorithm"], "sha256_cbor")
        self.assertEqual(config["qat"]["module_name"], "kv_agent")
        self.assertEqual(config["qat"]["instance_num"], 3)
        self.assertEqual(config["qat"]["loss_level"], 1)
        self.assertEqual(config["qat"]["log_enabled"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
