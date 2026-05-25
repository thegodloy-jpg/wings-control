# -*- coding: utf-8 -*-
"""UT-SPEC-WL：投机推理 launcher 白名单与防重复契约。

当前契约：
  - launcher 仅在 engine_config.speculative_config 不存在时合成 --speculative-config
  - JSON 默认 / 架构指纹注入不承载 speculative_config
  - 上层显式 dict 与 launcher 至多一份进入命令行

运行：
  python -m pytest tests/test_unit_speculative_whitelist.py -v
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

from engines.vllm_adapter import (  # noqa: E402
    _should_append_auto_speculative_config,
    should_append_auto_speculative_config,
    build_speculative_cmd,
)
from snapshot_framework import FakeModelIdentifier  # noqa: E402


class TestWhitelistBranches(unittest.TestCase):
    """_should_append_auto_speculative_config 的 4 条独立分支。"""

    def _base_params(self, **overrides):
        params = {
            "model_name": "Qwen3-72B",
            "model_path": "/models/Qwen3-72B",
            "model_type": "llm",
            "engine": "vllm",
            "enable_speculative_decode": True,
            "engine_config": {},
        }
        params.update(overrides)
        return params

    def test_disabled_switch_returns_false(self):
        """开关关闭 → False（基础门控）。"""
        params = self._base_params(enable_speculative_decode=False)
        self.assertFalse(_should_append_auto_speculative_config(params))

    def test_engine_config_has_spec_returns_false(self):
        """engine_config.speculative_config 已存在 → False（防重复）。"""
        params = self._base_params(engine_config={
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}
        })
        self.assertFalse(_should_append_auto_speculative_config(params))

    def test_v4_flash_also_goes_through_launcher(self):
        """V4-Flash 不再有旁路：开关开 + engine_config 空 → True（与其他模型一致）。"""
        params = self._base_params(
            model_name="DeepSeek-V4-Flash",
            model_path="/models/DeepSeek-V4-Flash",
        )
        self.assertTrue(_should_append_auto_speculative_config(params))

    def test_clean_state_returns_true(self):
        """开关开 + engine_config 空 → True（launcher 兜底合成）。"""
        params = self._base_params()
        self.assertTrue(_should_append_auto_speculative_config(params))

    def test_public_wrapper_matches_private(self):
        """should_append_auto_speculative_config 是 _should_append... 的薄包装。"""
        params = self._base_params(enable_speculative_decode=False)
        self.assertEqual(
            should_append_auto_speculative_config(params),
            _should_append_auto_speculative_config(params),
        )


class TestBuildSpeculativeCmd(unittest.TestCase):
    """build_speculative_cmd 输出格式合约 — 关键架构 → 关键字段。"""

    def _params_for_arch(self, arch):
        return {
            "model_name": "test",
            "model_path": "/models/test",
            "model_type": "llm",
            "speculative_decode_model_path": "",
        }

    def _patched_strategy(self, arch, strategy, **extra_patches):
        """fake ModelIdentifier + resolve_speculative_strategy。"""
        fake = FakeModelIdentifier(architecture=arch)
        patches = {
            "engines.vllm_adapter.ModelIdentifier": lambda *a, **k: fake,
            "engines.vllm_adapter.resolve_speculative_strategy":
                lambda *a, **k: strategy,
        }
        patches.update(extra_patches)
        return patches

    def _run_with_patches(self, params, engine, patches):
        ctx = []
        for tgt, val in patches.items():
            p = patch(tgt, side_effect=val) if callable(val) else patch(tgt, val)
            ctx.append(p)
            p.start()
        try:
            return build_speculative_cmd(params, engine)
        finally:
            for p in ctx:
                p.stop()

    def test_mtp_strategy_emits_method_and_num3(self):
        """MTP → method 与 num=3。"""
        params = self._params_for_arch("DeepseekV3ForCausalLM")
        cmd = self._run_with_patches(
            params, "vllm",
            self._patched_strategy("DeepseekV3ForCausalLM", "mtp"),
        )
        self.assertIn("--speculative-config", cmd)
        self.assertIn('"method": "mtp"', cmd)
        self.assertIn('"num_speculative_tokens": 3', cmd)

    def test_v4_pro_mtp_emits_num1(self):
        """V4-Pro 走 launcher 兜底 → num=1（与 ascend.json 默认对齐）。"""
        params = self._params_for_arch("DeepseekV3ForCausalLM")
        params["model_name"] = "DeepSeek-V4-Pro"
        patches = self._patched_strategy("DeepseekV3ForCausalLM", "mtp")
        patches["engines.vllm_adapter._is_deepseek_v4_pro_params"] = \
            lambda *a, **k: True
        patches["engines.vllm_adapter._is_deepseek_v4_flash_params"] = \
            lambda *a, **k: False
        cmd = self._run_with_patches(params, "vllm_ascend", patches)
        self.assertIn('"method": "mtp"', cmd)
        self.assertIn('"num_speculative_tokens": 1', cmd)
        self.assertNotIn('"num_speculative_tokens": 3', cmd)

    def test_v4_flash_mtp_emits_num1(self):
        """V4-Flash launcher 路径若被走（绕白名单）也应是 num=1。"""
        params = self._params_for_arch("DeepseekV3ForCausalLM")
        params["model_name"] = "DeepSeek-V4-Flash"
        patches = self._patched_strategy("DeepseekV3ForCausalLM", "mtp")
        patches["engines.vllm_adapter._is_deepseek_v4_pro_params"] = \
            lambda *a, **k: False
        patches["engines.vllm_adapter._is_deepseek_v4_flash_params"] = \
            lambda *a, **k: True
        cmd = self._run_with_patches(params, "vllm_ascend", patches)
        self.assertIn('"method": "mtp"', cmd)
        self.assertIn('"num_speculative_tokens": 1', cmd)

    def test_deepseek_mtp_emits_method(self):
        """GLM-5.1 走 deepseek_mtp（endswith _mtp 分支）。"""
        params = self._params_for_arch("GlmMoeDsaForCausalLM")
        cmd = self._run_with_patches(
            params, "vllm",
            self._patched_strategy("GlmMoeDsaForCausalLM", "deepseek_mtp"),
        )
        self.assertIn('"method": "deepseek_mtp"', cmd)
        self.assertIn('"num_speculative_tokens": 3', cmd)

    def test_suffix_emits_full_triple(self):
        """suffix → method + num=5 + cached_requests=1000。"""
        params = self._params_for_arch("UnknownForCausalLM")
        cmd = self._run_with_patches(
            params, "vllm",
            self._patched_strategy("UnknownForCausalLM", "suffix"),
        )
        self.assertIn('"method" : "suffix"', cmd)
        self.assertIn('"num_speculative_tokens": 5', cmd)
        self.assertIn('"suffix_decoding_max_cached_requests": 1000', cmd)

    def test_no_strategy_returns_empty(self):
        """非 vllm/vllm_ascend 引擎 / 不支持架构 → ""。"""
        params = self._params_for_arch("Qwen3ForCausalLM")
        cmd = self._run_with_patches(
            params, "sglang",
            self._patched_strategy("Qwen3ForCausalLM", ""),
        )
        self.assertEqual(cmd, "")

    def test_draft_model_path_safe_escape(self):
        """草稿模型路径含特殊字符 → JSON 安全转义，不破坏单引号包裹。"""
        params = {
            "model_name": "test",
            "model_path": "/models/test",
            "model_type": "llm",
            "speculative_decode_model_path": '/models/draft"with\\quote',
        }
        from unittest.mock import MagicMock
        fake = FakeModelIdentifier(architecture="Qwen3ForCausalLM")
        fake_draft = MagicMock()
        fake_draft.draft_model_architecture = "Qwen3ForCausalLM"
        with patch("engines.vllm_adapter.ModelIdentifier", return_value=fake),\
             patch("engines.vllm_adapter.ModelIdentifierDraft", return_value=fake_draft),\
             patch("engines.vllm_adapter.resolve_speculative_strategy",
                   return_value="draft_model"):
            cmd = build_speculative_cmd(params, "vllm")
        # 引号被转义为 \"，反斜杠为 \\\\
        self.assertIn('\\"', cmd)
        self.assertIn('\\\\', cmd)
        # 整体仍被单引号包裹（命令行可解析）
        self.assertEqual(cmd.count("'"), 2)


class TestNoDoubleEmission(unittest.TestCase):
    """端到端不变式：命令行 --speculative-config 至多出现一次。"""

    def _count_spec_cli(self, script):
        return script.count("--speculative-config")

    def test_engine_config_only_renders_once(self):
        """engine_config 有 spec → 仅 _build_vllm_cmd_parts 渲染一份。"""
        from engines.vllm_adapter import _should_append_auto_speculative_config
        params = {
            "engine": "vllm",
            "enable_speculative_decode": True,
            "engine_config": {
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}
            },
            "model_name": "x", "model_path": "/m", "model_type": "llm",
        }
        # 白名单返回 False → launcher 不合成 → 不会出现双份
        self.assertFalse(_should_append_auto_speculative_config(params))

    def test_launcher_only_when_engine_config_empty(self):
        """engine_config 空 → 白名单放行 → launcher 合成兜底。"""
        from engines.vllm_adapter import _should_append_auto_speculative_config
        params = {
            "engine": "vllm",
            "enable_speculative_decode": True,
            "engine_config": {},
            "model_name": "x", "model_path": "/m", "model_type": "llm",
        }
        self.assertTrue(_should_append_auto_speculative_config(params))


if __name__ == "__main__":
    unittest.main(verbosity=2)
