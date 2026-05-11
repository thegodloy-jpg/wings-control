# -*- coding: utf-8 -*-
"""UT-ACCEL：Accel 特性收集逻辑单元测试。

覆盖 TEST_PLAN.md GAP-1 中识别的缺失覆盖范围：
  - _dedupe_features
  - _merge_patch_options
  - _validate_accel_user_override
  - _collect_ears_patch_features（EARS 仅 vllm，非 vllm_ascend）
  - _collect_indexcache_patch_features（IndexCache 仅 GlmMoeDsa / DeepseekV32）

运行：
  python -m pytest tests/test_unit_accel.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from core.wings_entry import (  # noqa: E402
    _dedupe_features,
    _merge_patch_options,
    _validate_accel_user_override,
    _collect_ears_patch_features,
    _collect_indexcache_patch_features,
)
from snapshot_framework import FakeModelIdentifier  # noqa: E402


# ---------------------------------------------------------------------------
# _dedupe_features
# ---------------------------------------------------------------------------

class TestDedupeFeatures(unittest.TestCase):
    """验证 _dedupe_features 的去重和顺序保持。"""

    def test_no_duplicates_preserved_in_order(self):
        self.assertEqual(_dedupe_features(["a", "b", "c"]), ["a", "b", "c"])

    def test_duplicates_removed_first_seen_wins(self):
        self.assertEqual(_dedupe_features(["a", "b", "a", "c"]), ["a", "b", "c"])

    def test_empty_strings_excluded(self):
        self.assertEqual(_dedupe_features(["", "a", ""]), ["a"])

    def test_empty_list(self):
        self.assertEqual(_dedupe_features([]), [])

    def test_all_same(self):
        self.assertEqual(_dedupe_features(["x", "x", "x"]), ["x"])


# ---------------------------------------------------------------------------
# _merge_patch_options
# ---------------------------------------------------------------------------

class TestMergePatchOptions(unittest.TestCase):
    """验证 _merge_patch_options 的 JSON 合并逻辑。"""

    def test_empty_raw_creates_new_options(self):
        result = _merge_patch_options("", "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        self.assertEqual(parsed["vllm"]["features"], ["ears"])
        self.assertEqual(parsed["vllm"]["version"], "1.0.0")

    def test_existing_features_merged_without_dup(self):
        existing = json.dumps({"vllm": {"version": "1.0.0", "features": ["indexcache"]}})
        result = _merge_patch_options(existing, "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        features = parsed["vllm"]["features"]
        self.assertIn("indexcache", features)
        self.assertIn("ears", features)
        self.assertEqual(len(features), 2)

    def test_duplicate_features_deduped(self):
        existing = json.dumps({"vllm": {"version": "1.0.0", "features": ["ears"]}})
        result = _merge_patch_options(existing, "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        self.assertEqual(parsed["vllm"]["features"], ["ears"])

    def test_version_preserved_if_exists(self):
        existing = json.dumps({"vllm": {"version": "2.0.0", "features": []}})
        result = _merge_patch_options(existing, "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        self.assertEqual(parsed["vllm"]["version"], "2.0.0")

    def test_invalid_json_treated_as_empty(self):
        result = _merge_patch_options("NOT_JSON", "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        self.assertEqual(parsed["vllm"]["features"], ["ears"])

    def test_other_engine_key_untouched(self):
        existing = json.dumps({"sglang": {"version": "1.0.0", "features": ["x"]}})
        result = _merge_patch_options(existing, "vllm", "1.0.0", ["ears"])
        parsed = json.loads(result)
        self.assertIn("sglang", parsed)
        self.assertIn("vllm", parsed)


# ---------------------------------------------------------------------------
# _validate_accel_user_override
# ---------------------------------------------------------------------------

class TestValidateAccelUserOverride(unittest.TestCase):
    """验证 _validate_accel_user_override 对 WINGS_ENGINE_PATCH_OPTIONS 的校验。"""

    def test_empty_env_returns_empty(self):
        with patch.dict(os.environ, {"WINGS_ENGINE_PATCH_OPTIONS": ""}, clear=False):
            self.assertEqual(_validate_accel_user_override(), "")

    def test_valid_json_dict_returned(self):
        payload = json.dumps({"vllm": {"version": "1.0.0", "features": ["ears"]}})
        with patch.dict(os.environ, {"WINGS_ENGINE_PATCH_OPTIONS": payload}, clear=False):
            result = _validate_accel_user_override()
        self.assertEqual(result, payload)

    def test_invalid_json_returns_empty(self):
        with patch.dict(os.environ, {"WINGS_ENGINE_PATCH_OPTIONS": "INVALID"}, clear=False):
            self.assertEqual(_validate_accel_user_override(), "")

    def test_non_dict_json_returns_empty(self):
        with patch.dict(os.environ, {"WINGS_ENGINE_PATCH_OPTIONS": "[1,2,3]"}, clear=False):
            self.assertEqual(_validate_accel_user_override(), "")

    def test_whitespace_only_returns_empty(self):
        with patch.dict(os.environ, {"WINGS_ENGINE_PATCH_OPTIONS": "   "}, clear=False):
            self.assertEqual(_validate_accel_user_override(), "")


# ---------------------------------------------------------------------------
# _collect_ears_patch_features
# ---------------------------------------------------------------------------

class TestCollectEarsFeatures(unittest.TestCase):
    """验证 EARS 补丁特性收集逻辑（仅 vllm，不含 vllm_ascend）。"""

    def _ears(self, engine, spec_enabled, spec_model_path="", strategy_return="mtp"):
        merged = {
            "enable_speculative_decode": spec_enabled,
            "speculative_decode_model_path": spec_model_path,
            "model_name": "test-model",
            "model_path": "/models/test",
        }
        with patch("core.wings_entry.resolve_speculative_strategy",
                   return_value=strategy_return):
            return _collect_ears_patch_features(engine, merged)

    def test_vllm_spec_enabled_mtp_strategy_returns_ears(self):
        """vllm + spec enabled + MTP strategy → ['ears']。"""
        self.assertEqual(self._ears("vllm", True, strategy_return="mtp"), ["ears"])

    def test_vllm_spec_enabled_suffix_strategy_returns_ears(self):
        """vllm + spec enabled + suffix strategy → ['ears']。"""
        self.assertEqual(self._ears("vllm", True, strategy_return="suffix"), ["ears"])

    def test_vllm_spec_enabled_deepseek_mtp_returns_ears(self):
        """vllm + spec enabled + deepseek_mtp strategy → ['ears']（ends with _mtp）。"""
        self.assertEqual(self._ears("vllm", True, strategy_return="deepseek_mtp"), ["ears"])

    def test_vllm_ascend_spec_enabled_no_ears(self):
        """vllm_ascend + spec enabled → [] (Ascend 不安装 EARS 补丁)。"""
        self.assertEqual(self._ears("vllm_ascend", True, strategy_return="mtp"), [])

    def test_vllm_spec_disabled_no_ears(self):
        """vllm + spec disabled → []。"""
        self.assertEqual(self._ears("vllm", False), [])

    def test_vllm_with_draft_model_no_ears(self):
        """vllm + draft model path 存在 → strategy=draft_model，不触发 EARS。"""
        merged = {
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "/models/draft",
            "model_name": "test-model",
            "model_path": "/models/test",
        }
        result = _collect_ears_patch_features("vllm", merged)
        self.assertEqual(result, [])

    def test_sglang_no_ears(self):
        """sglang 引擎 → [] (非 vllm 引擎不触发)。"""
        self.assertEqual(self._ears("sglang", True), [])


# ---------------------------------------------------------------------------
# _collect_indexcache_patch_features
# ---------------------------------------------------------------------------

class TestCollectIndexcacheFeatures(unittest.TestCase):
    """验证 IndexCache 特性收集逻辑（仅 vllm + GlmMoeDsa/DeepseekV32）。"""

    def _ic(self, engine, arch, enable_sparse=True):
        fake_model = FakeModelIdentifier(architecture=arch)

        def _make_fake(name, path, mtype):
            return fake_model

        merged = {
            "enable_sparse": enable_sparse,
            "model_name": "test-model",
            "model_path": "/models/test",
        }
        with patch("core.wings_entry.ModelIdentifier", side_effect=_make_fake):
            return _collect_indexcache_patch_features(engine, merged)

    def test_vllm_glm_moe_dsa_returns_indexcache(self):
        """vllm + GlmMoeDsaForCausalLM + enable_sparse → ['indexcache']。"""
        self.assertEqual(self._ic("vllm", "GlmMoeDsaForCausalLM"), ["indexcache"])

    def test_vllm_deepseekv32_returns_indexcache(self):
        """vllm + DeepseekV32ForCausalLM + enable_sparse → ['indexcache']。"""
        self.assertEqual(self._ic("vllm", "DeepseekV32ForCausalLM"), ["indexcache"])

    def test_vllm_llama_non_indexcache_arch_returns_empty(self):
        """vllm + LlamaForCausalLM（非 IndexCache 架构）→ []（走 FP8 路径，无需补丁）。"""
        self.assertEqual(self._ic("vllm", "LlamaForCausalLM"), [])

    def test_vllm_qwen3_non_indexcache_returns_empty(self):
        """vllm + Qwen3ForCausalLM → []（非 IndexCache 架构）。"""
        self.assertEqual(self._ic("vllm", "Qwen3ForCausalLM"), [])

    def test_vllm_ascend_glm_moe_dsa_returns_empty(self):
        """vllm_ascend + GlmMoeDsaForCausalLM → []（IndexCache 仅支持 vllm 引擎）。"""
        self.assertEqual(self._ic("vllm_ascend", "GlmMoeDsaForCausalLM"), [])

    def test_enable_sparse_false_returns_empty(self):
        """enable_sparse=False → [] (开关未开启)。"""
        self.assertEqual(self._ic("vllm", "GlmMoeDsaForCausalLM", enable_sparse=False), [])

    def test_sglang_glm_moe_dsa_returns_empty(self):
        """sglang + GlmMoeDsaForCausalLM → []（非 vllm 引擎）。"""
        self.assertEqual(self._ic("sglang", "GlmMoeDsaForCausalLM"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
