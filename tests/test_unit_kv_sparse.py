# -*- coding: utf-8 -*-
"""UT-KVS：KV 稀疏策略（_build_kv_sparse_cmd）单元测试。

覆盖 TEST_PLAN.md GAP-3：
  - IndexCache 架构（GlmMoeDsa / DeepseekV32）→ --hf-overrides 参数
  - 其他架构（LlamaForCausalLM 等）→ 就地注入 kv_cache_dtype=fp8 / calculate_kv_scales=True
  - 非 vllm 引擎（vllm_ascend / sglang）→ 返回空字符串，不修改 params

运行：
  python -m pytest tests/test_unit_kv_sparse.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from engines.vllm_adapter import _build_kv_sparse_cmd  # noqa: E402
from snapshot_framework import FakeModelIdentifier  # noqa: E402


def _make_params(arch: str, engine_config: dict | None = None) -> dict:
    params = {
        "model_name": "test-model",
        "model_path": "/models/test",
        "model_type": "llm",
    }
    if engine_config is not None:
        params["engine_config"] = engine_config
    return params


def _fake_model(arch: str):
    def _side_effect(name, path, mtype):
        return FakeModelIdentifier(architecture=arch)
    return _side_effect


class TestBuildKvSparseCmd(unittest.TestCase):
    """验证 _build_kv_sparse_cmd 的双路径逻辑和引擎门控。"""

    # ── IndexCache 路径 ──────────────────────────────────────────────────────

    def test_glm_moe_dsa_returns_hf_overrides(self):
        """GlmMoeDsaForCausalLM + vllm → 返回 --hf-overrides 参数字符串。"""
        params = _make_params("GlmMoeDsaForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("GlmMoeDsaForCausalLM")):
            result = _build_kv_sparse_cmd(params, "vllm")
        self.assertIn("--hf-overrides", result)
        self.assertIn("index_topk_freq", result)
        self.assertIn("4", result)

    def test_deepseekv32_returns_hf_overrides(self):
        """DeepseekV32ForCausalLM + vllm → 返回 --hf-overrides 参数字符串。"""
        params = _make_params("DeepseekV32ForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("DeepseekV32ForCausalLM")):
            result = _build_kv_sparse_cmd(params, "vllm")
        self.assertIn("--hf-overrides", result)
        self.assertIn("index_topk_freq", result)

    def test_indexcache_does_not_modify_engine_config(self):
        """IndexCache 路径不修改 engine_config（参数通过 CLI 传递，不写入配置）。"""
        params = _make_params("GlmMoeDsaForCausalLM", engine_config={})
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("GlmMoeDsaForCausalLM")):
            _build_kv_sparse_cmd(params, "vllm")
        self.assertNotIn("kv_cache_dtype", params.get("engine_config", {}))

    # ── FP8 KV CACHE 路径 ───────────────────────────────────────────────────

    def test_llama_injects_fp8_kv_cache_dtype(self):
        """LlamaForCausalLM + vllm → 就地注入 kv_cache_dtype=fp8。"""
        params = _make_params("LlamaForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("LlamaForCausalLM")):
            result = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(result, "")
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")

    def test_llama_injects_calculate_kv_scales(self):
        """LlamaForCausalLM + vllm → 就地注入 calculate_kv_scales=True。"""
        params = _make_params("LlamaForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("LlamaForCausalLM")):
            _build_kv_sparse_cmd(params, "vllm")
        self.assertTrue(params["engine_config"]["calculate_kv_scales"])

    def test_qwen3_injects_fp8_returns_empty(self):
        """Qwen3ForCausalLM + vllm → FP8 路径，返回空字符串。"""
        params = _make_params("Qwen3ForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("Qwen3ForCausalLM")):
            result = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(result, "")
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")

    def test_deepseekv3_injects_fp8_returns_empty(self):
        """DeepseekV3ForCausalLM + vllm → FP8 路径（非 V32，无 IndexCache）。"""
        params = _make_params("DeepseekV3ForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("DeepseekV3ForCausalLM")):
            result = _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(result, "")
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")

    def test_fp8_preserves_existing_engine_config_keys(self):
        """FP8 路径就地追加，不破坏 engine_config 已有的其他键值对。"""
        params = _make_params("LlamaForCausalLM",
                              engine_config={"tensor_parallel_size": 4})
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=_fake_model("LlamaForCausalLM")):
            _build_kv_sparse_cmd(params, "vllm")
        self.assertEqual(params["engine_config"]["tensor_parallel_size"], 4)
        self.assertEqual(params["engine_config"]["kv_cache_dtype"], "fp8")

    # ── 引擎门控 ─────────────────────────────────────────────────────────────

    def test_vllm_ascend_returns_empty_no_modification(self):
        """vllm_ascend 引擎 → 返回空字符串，不修改 params（KV Sparse 仅限 NVIDIA vllm）。"""
        params = _make_params("GlmMoeDsaForCausalLM")
        result = _build_kv_sparse_cmd(params, "vllm_ascend")
        self.assertEqual(result, "")
        self.assertNotIn("engine_config", params)

    def test_sglang_returns_empty_no_modification(self):
        """sglang 引擎 → 返回空字符串，不修改 params。"""
        params = _make_params("DeepseekV3ForCausalLM")
        result = _build_kv_sparse_cmd(params, "sglang")
        self.assertEqual(result, "")
        self.assertNotIn("engine_config", params)

    def test_mindie_returns_empty_no_modification(self):
        """mindie 引擎 → 返回空字符串，不修改 params。"""
        params = _make_params("LlamaForCausalLM")
        result = _build_kv_sparse_cmd(params, "mindie")
        self.assertEqual(result, "")
        self.assertNotIn("engine_config", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
