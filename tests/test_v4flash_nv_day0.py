# -*- coding: utf-8 -*-
"""[V4-Flash-NV-Day0] DeepSeek-V4-Flash · NVIDIA Day0 特性适配单测。

覆盖三项特性 + 互斥 + fallback：
  D1 投机 method：NV → mtp；Ascend → deepseek_mtp（不回归）
  D2 IndexCache：NV V4-Flash → use_index_cache，且不触发 indexcache 补丁
  D3/D4 native KV 卸载：复用 LMCACHE_OFFLOAD；size = device_count × 每卡值（默认 200）
  互斥：native 卸载命中 → 不导出 LMCache engine 侧 env
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(ROOT))

from wings_control.engines import vllm_adapter  # noqa: E402


class _FakeModelInfo:
    def __init__(self, architecture: str):
        self.model_architecture = architecture
        self.model_quantize = ""


def _v4_flash_params(**overrides):
    params = {
        "model_name": "DeepSeek-V4-Flash",
        "model_path": "/models/DeepSeek-V4-Flash",
        "model_type": "llm",
        "engine": "vllm",
        "device_count": 8,
        "engine_config": {},
    }
    params.update(overrides)
    return params


class TestSpeculativeMethod(unittest.TestCase):
    """D1：method 字面值按 engine 收口。"""

    def test_nv_v4_flash_method_is_mtp(self):
        params = _v4_flash_params(enable_speculative_decode=True)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {}, clear=False):
            cmd = vllm_adapter.build_speculative_cmd(params, "vllm")
        self.assertIn('"method": "mtp"', cmd)
        self.assertNotIn('deepseek_mtp', cmd)
        self.assertIn('"num_speculative_tokens": 1', cmd)

    def test_ascend_v4_flash_keeps_deepseek_mtp(self):
        """Ascend V4-Flash 维持官方模板 deepseek_mtp，不被 NV 覆盖。"""
        params = _v4_flash_params(engine="vllm_ascend", enable_speculative_decode=True)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")):
            cmd = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
        self.assertIn('"method": "deepseek_mtp"', cmd)


class TestIndexCache(unittest.TestCase):
    """D2：NV V4-Flash IndexCache use_index_cache，无补丁。"""

    def test_nv_v4_flash_emits_use_index_cache(self):
        params = _v4_flash_params()
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")):
            extra = vllm_adapter._build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(
            extra,
            " --hf-overrides '{\"use_index_cache\": true, \"index_topk_freq\": 4}'",
        )
        # 走 IndexCache 分支 → 不再注入 FP8 kv_cache_dtype（fp8 由基础配置提供）
        self.assertNotIn("kv_cache_dtype", params["engine_config"])

    def test_no_indexcache_patch_for_v4_flash(self):
        """D2：V4 不在 INDEXCACHE_ARCHS → 补丁聚合天然返回 []（不装补丁）。"""
        from core.wings_entry import _collect_indexcache_patch_features  # noqa: E402
        merged = _v4_flash_params(enable_sparse=True)
        with patch("core.wings_entry.ModelIdentifier",
                   return_value=_FakeModelInfo("DeepseekV4ForCausalLM")):
            feats = _collect_indexcache_patch_features("vllm", merged)
        self.assertEqual(feats, [])


class TestSparseSwitchGating(unittest.TestCase):
    """端到端：NV V4-Flash IndexCache 默认强制开（方案 A，绕过 enable_sparse，关不掉）。"""

    def test_sparse_on_emits_hf_overrides(self):
        params = _v4_flash_params(enable_sparse=True)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {}, clear=True):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("use_index_cache", script)

    def test_sparse_off_still_emits_hf_overrides(self):
        """显式 enable_sparse=False 也关不掉（与 GLM-5.1-Ascend 同范式：强制开）。"""
        params = _v4_flash_params(enable_sparse=False)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {}, clear=True):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("use_index_cache", script)

    def test_sparse_missing_emits_hf_overrides(self):
        """未传 enable_sparse → 默认开。"""
        params = _v4_flash_params()  # 未传 enable_sparse
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {}, clear=True):
            script = vllm_adapter.build_start_script(params)
        self.assertIn("use_index_cache", script)


class TestNativeKvOffload(unittest.TestCase):
    """D3/D4：native 卸载触发与容量乘法。"""

    def test_size_multiplies_by_device_count(self):
        params = _v4_flash_params(device_count=8)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ",
                        {"LMCACHE_OFFLOAD": "true", "LMCACHE_MAX_LOCAL_CPU_SIZE": "25"},
                        clear=False):
            cmd = vllm_adapter._build_kv_offload_cmd(params, "vllm")
        self.assertEqual(cmd, " --kv_offloading_backend native --kv_offloading_size 200")

    def test_default_200_when_env_unset(self):
        params = _v4_flash_params(device_count=8)
        env = {"LMCACHE_OFFLOAD": "true"}
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", env, clear=True):
            cmd = vllm_adapter._build_kv_offload_cmd(params, "vllm")
        self.assertEqual(cmd, " --kv_offloading_backend native --kv_offloading_size 200")

    def test_disabled_when_lmcache_off(self):
        params = _v4_flash_params(device_count=8)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {}, clear=True):
            cmd = vllm_adapter._build_kv_offload_cmd(params, "vllm")
        self.assertEqual(cmd, "")

    def test_fallback_flag_suppresses_offload(self):
        params = _v4_flash_params(device_count=8, _wings_fallback_no_kv_offload=True)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {"LMCACHE_OFFLOAD": "true"}, clear=True):
            cmd = vllm_adapter._build_kv_offload_cmd(params, "vllm")
        self.assertEqual(cmd, "")

    def test_not_applied_on_ascend(self):
        params = _v4_flash_params(engine="vllm_ascend", device_count=8)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {"LMCACHE_OFFLOAD": "true"}, clear=True):
            cmd = vllm_adapter._build_kv_offload_cmd(params, "vllm_ascend")
        self.assertEqual(cmd, "")


class TestOffloadLmcacheMutualExclusion(unittest.TestCase):
    """互斥：NV V4-Flash native 卸载命中 → 跳过 LMCache engine 侧 env。"""

    def test_nv_v4_flash_skips_lmcache_env(self):
        params = _v4_flash_params(device_count=8)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {"LMCACHE_OFFLOAD": "true"}, clear=True):
            env_cmds = vllm_adapter._build_cache_env_commands("vllm", params)
        joined = "\n".join(env_cmds)
        self.assertNotIn("LMCACHE_OFFLOAD", joined)
        self.assertNotIn("LMCACHE_CONFIG_FILE", joined)


class TestSharedOffloadGbHelper(unittest.TestCase):
    """ascend cpu_swap_space_gb 与 NV native size 同源同值。"""

    def test_v4_flash_multiplies(self):
        params = _v4_flash_params(device_count=8)
        with patch.object(vllm_adapter, "ModelIdentifier",
                          return_value=_FakeModelInfo("DeepseekV4ForCausalLM")), \
             patch.dict("os.environ", {"LMCACHE_MAX_LOCAL_CPU_SIZE": "25"}, clear=True):
            self.assertEqual(vllm_adapter._resolve_v4_flash_offload_gb(params), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
