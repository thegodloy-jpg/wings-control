# -*- coding: utf-8 -*-
"""UT-SPEC：投机推理策略解析单元测试。

覆盖 TEST_PLAN.md GAP-2 中识别的缺失范围：
  - _resolve_mtp_method：7 个架构 → MTP 方法名映射
  - resolve_speculative_strategy：5 条独立分支

运行：
  python -m pytest tests/test_unit_speculative.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from engines.vllm_adapter import _resolve_mtp_method, resolve_speculative_strategy  # noqa: E402
from snapshot_framework import FakeModelIdentifier  # noqa: E402


# ---------------------------------------------------------------------------
# _resolve_mtp_method
# ---------------------------------------------------------------------------

class TestResolveMtpMethod(unittest.TestCase):
    """验证 _resolve_mtp_method 的架构→MTP方法名映射（7 个已知 + 未知回退）。"""

    def test_deepseekv3_mtp(self):
        self.assertEqual(_resolve_mtp_method("DeepseekV3ForCausalLM"), "mtp")

    def test_deepseekv32_mtp(self):
        self.assertEqual(_resolve_mtp_method("DeepseekV32ForCausalLM"), "mtp")

    def test_glm_moe_dsa_deepseek_mtp(self):
        self.assertEqual(_resolve_mtp_method("GlmMoeDsaForCausalLM"), "deepseek_mtp")

    def test_qwen3next_qwen3_next_mtp(self):
        self.assertEqual(_resolve_mtp_method("Qwen3NextForCausalLM"), "qwen3_next_mtp")

    def test_glm4moe_glm4_moe_mtp(self):
        self.assertEqual(_resolve_mtp_method("Glm4MoeForCausalLM"), "glm4_moe_mtp")

    def test_qwen3_5_qwen3_5_mtp(self):
        self.assertEqual(_resolve_mtp_method("Qwen3_5ForConditionalGeneration"), "qwen3_5_mtp")

    def test_qwen3_5_moe_qwen3_5_mtp(self):
        self.assertEqual(_resolve_mtp_method("Qwen3_5MoeForConditionalGeneration"), "qwen3_5_mtp")

    def test_unknown_arch_empty_string(self):
        """未知架构 → 空字符串（caller 将选择 suffix 策略）。"""
        self.assertEqual(_resolve_mtp_method("UnknownArchForCausalLM"), "")

    def test_llama_no_mtp(self):
        """LlamaForCausalLM 无 MTP 支持 → 空字符串。"""
        self.assertEqual(_resolve_mtp_method("LlamaForCausalLM"), "")

    def test_qwen3_no_mtp(self):
        """Qwen3ForCausalLM 无 MTP 支持（非 Qwen3Next）→ 空字符串。"""
        self.assertEqual(_resolve_mtp_method("Qwen3ForCausalLM"), "")


# ---------------------------------------------------------------------------
# resolve_speculative_strategy
# ---------------------------------------------------------------------------

class TestResolveSpeculativeStrategy(unittest.TestCase):
    """验证 resolve_speculative_strategy 的 5 条独立分支。"""

    def _make_params(self, arch="Qwen3ForCausalLM", spec_model_path=""):
        return {
            "model_name": "test-model",
            "model_path": "/models/test",
            "model_type": "llm",
            "speculative_decode_model_path": spec_model_path,
            # 本套用例测 arch→投机策略映射，与白名单成员资格正交。灌入收口产物
            # _smart_feats（模拟 C14 已判定该模型允许 spec），让 §2.3 gate 放行专注测 arch。
            "_smart_feats": ["spec", "sparse", "offload"],
        }

    def _fake_model(self, arch):
        fake = FakeModelIdentifier(architecture=arch)

        def _side_effect(name, path, mtype):
            return fake

        return _side_effect

    # 分支1：非 vllm/vllm_ascend 引擎 → 空字符串
    def test_sglang_engine_returns_empty(self):
        result = resolve_speculative_strategy(self._make_params(), "sglang")
        self.assertEqual(result, "")

    def test_mindie_engine_returns_empty(self):
        result = resolve_speculative_strategy(self._make_params(), "mindie")
        self.assertEqual(result, "")

    # 分支2：spec_model_path 存在 + 普通草稿模型 → "draft_model"
    def test_draft_model_path_returns_draft_model(self):
        params = self._make_params(spec_model_path="/models/Qwen3-0.6B")
        fake_draft = MagicMock()
        fake_draft.draft_model_architecture = "Qwen3ForCausalLM"
        with patch("engines.vllm_adapter.ModelIdentifierDraft", return_value=fake_draft):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "draft_model")

    # 分支3：spec_model_path + eagle3 架构 → "eagle3"
    def test_eagle3_draft_model_returns_eagle3(self):
        params = self._make_params(spec_model_path="/models/Eagle3")
        fake_draft = MagicMock()
        fake_draft.draft_model_architecture = "Eagle3ForCausalLM"
        with patch("engines.vllm_adapter.ModelIdentifierDraft", return_value=fake_draft):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "eagle3")

    # 分支4：Qwen3NextForCausalLM + vllm_ascend → "suffix"
    def test_qwen3next_vllm_ascend_returns_suffix(self):
        params = self._make_params(arch="Qwen3NextForCausalLM")
        with patch("engines.vllm_adapter.ModelIdentifier",
                   side_effect=self._fake_model("Qwen3NextForCausalLM")):
            result = resolve_speculative_strategy(params, "vllm_ascend")
        self.assertEqual(result, "suffix")

    # 分支5a：MTP 架构 + LMCACHE_OFFLOAD=false → MTP method
    def test_deepseekv3_vllm_no_lmcache_returns_mtp(self):
        params = self._make_params(arch="DeepseekV3ForCausalLM")
        with (
            patch("engines.vllm_adapter.ModelIdentifier",
                  side_effect=self._fake_model("DeepseekV3ForCausalLM")),
            patch("engines.vllm_adapter.get_lmcache_env", return_value=False),
        ):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "mtp")

    # 分支5b：MTP 架构 + LMCACHE_OFFLOAD=true → "suffix"（lmcache 与 MTP 不兼容）
    def test_deepseekv3_vllm_with_lmcache_returns_suffix(self):
        params = self._make_params(arch="DeepseekV3ForCausalLM")
        with (
            patch("engines.vllm_adapter.ModelIdentifier",
                  side_effect=self._fake_model("DeepseekV3ForCausalLM")),
            patch("engines.vllm_adapter.get_lmcache_env", return_value=True),
        ):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "suffix")

    # 未知架构（无 MTP 方法）→ 兜底 "suffix"
    def test_unknown_arch_vllm_returns_suffix(self):
        params = self._make_params(arch="UnknownArchForCausalLM")
        with (
            patch("engines.vllm_adapter.ModelIdentifier",
                  side_effect=self._fake_model("UnknownArchForCausalLM")),
            patch("engines.vllm_adapter.get_lmcache_env", return_value=False),
        ):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "suffix")

    # GlmMoeDsa → deepseek_mtp (vllm, no lmcache)
    def test_glm_moe_dsa_returns_deepseek_mtp(self):
        params = self._make_params(arch="GlmMoeDsaForCausalLM")
        with (
            patch("engines.vllm_adapter.ModelIdentifier",
                  side_effect=self._fake_model("GlmMoeDsaForCausalLM")),
            patch("engines.vllm_adapter.get_lmcache_env", return_value=False),
        ):
            result = resolve_speculative_strategy(params, "vllm")
        self.assertEqual(result, "deepseek_mtp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
