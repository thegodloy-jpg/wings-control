# -*- coding: utf-8 -*-
"""UT-MFC：MindIE Function Call 配置注入单元测试。

覆盖 TEST_PLAN.md GAP-7（UT-TCP-25~29）：
  - _inject_function_call_config（mindie_adapter）：5 个注入分支
  - _set_mindie_function_call（config_loader）：enable_auto_tool_choice 门控

运行：
  python -m pytest tests/test_unit_mindie_fc.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.mindie_adapter import _inject_function_call_config  # noqa: E402
from core.config_loader import _set_mindie_function_call  # noqa: E402


# ---------------------------------------------------------------------------
# _inject_function_call_config（mindie_adapter）
# ---------------------------------------------------------------------------

class TestInjectFunctionCallConfig(unittest.TestCase):
    """验证 MindIE tool_call_options 注入逻辑（5 个分支）。"""

    # UT-TCP-25：无 mindie_tool_call_parser → 不注入
    def test_tcp25_no_parser_key_no_injection(self):
        """engine_config 中没有 mindie_tool_call_parser → overrides 不变。"""
        engine_config = {"mindie_model_type": "deepseekv2"}
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertEqual(overrides, {})

    # UT-TCP-26：无 mindie_model_type → 不注入
    def test_tcp26_no_model_type_no_injection(self):
        """engine_config 中没有 mindie_model_type → overrides 不变。"""
        engine_config = {"mindie_tool_call_parser": "deepseek_v31"}
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertEqual(overrides, {})

    # UT-TCP-27：parser != "deepseek_v31" → 跳过注入（只有 DeepSeek-V3.1 需要 tool_call_options）
    def test_tcp27_non_deepseek_v31_parser_skipped(self):
        """parser=deepseek_v3（非 v31）→ 不写入 tool_call_options。"""
        engine_config = {
            "mindie_tool_call_parser": "deepseek_v3",
            "mindie_model_type": "deepseekv2",
        }
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertNotIn("models", overrides)

    def test_tcp27b_qwen_parser_skipped(self):
        """parser=qwen（非 deepseek_v31）→ 不写入 tool_call_options。"""
        engine_config = {
            "mindie_tool_call_parser": "qwen",
            "mindie_model_type": "qwen",
        }
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertNotIn("models", overrides)

    # UT-TCP-28：parser == "deepseek_v31" → 注入 tool_call_options
    def test_tcp28_deepseek_v31_parser_injects_tool_call_options(self):
        """parser=deepseek_v31 + model_type=deepseekv2 → 注入 models.deepseekv2.tool_call_options。"""
        engine_config = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
        }
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertIn("models", overrides)
        self.assertIn("deepseekv2", overrides["models"])
        tool_opts = overrides["models"]["deepseekv2"]["tool_call_options"]
        self.assertEqual(tool_opts["tool_call_parser"], "deepseek_v31")

    # UT-TCP-29：parser == "deepseek_v31" + chat_template → 同时注入 chat_template
    def test_tcp29_deepseek_v31_with_chat_template_injected(self):
        """parser=deepseek_v31 + mindie_chat_template → 同时注入 chat_template 字段。"""
        engine_config = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
            "mindie_chat_template": "/templates/deepseek_v31.jinja",
        }
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        model_entry = overrides["models"]["deepseekv2"]
        self.assertEqual(model_entry["tool_call_options"]["tool_call_parser"], "deepseek_v31")
        self.assertEqual(model_entry["chat_template"], "/templates/deepseek_v31.jinja")

    def test_deepseek_v31_no_chat_template_no_extra_key(self):
        """无 mindie_chat_template → model_entry 中不存在 chat_template 键。"""
        engine_config = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
        }
        overrides = {}
        _inject_function_call_config(engine_config, overrides)
        self.assertNotIn("chat_template", overrides["models"]["deepseekv2"])

    def test_existing_overrides_models_dict_preserved(self):
        """overrides 中已有 models 结构 → setdefault 不覆盖现有内容。"""
        engine_config = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
        }
        overrides = {"models": {"llama": {"some_key": "some_val"}}}
        _inject_function_call_config(engine_config, overrides)
        self.assertIn("llama", overrides["models"])
        self.assertIn("deepseekv2", overrides["models"])


# ---------------------------------------------------------------------------
# _set_mindie_function_call（config_loader）
# ---------------------------------------------------------------------------

class TestSetMindieFunctionCall(unittest.TestCase):
    """验证 enable_auto_tool_choice 门控对 MindIE parser 字段的处理。"""

    # FC 开启 + parser 存在 → 保留所有字段
    def test_fc_enabled_with_parser_keeps_fields(self):
        params = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
        }
        _set_mindie_function_call(params, {"enable_auto_tool_choice": True})
        self.assertEqual(params["mindie_tool_call_parser"], "deepseek_v31")
        self.assertEqual(params["mindie_model_type"], "deepseekv2")

    # FC 开启 + parser 不存在 → 不抛异常，字段也不会被添加
    def test_fc_enabled_without_parser_no_error(self):
        params = {}
        _set_mindie_function_call(params, {"enable_auto_tool_choice": True})
        self.assertNotIn("mindie_tool_call_parser", params)

    # FC 关闭 → 清除 mindie_tool_call_parser / mindie_model_type / mindie_chat_template
    def test_fc_disabled_removes_parser_fields(self):
        params = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
            "mindie_chat_template": "/templates/foo.jinja",
        }
        _set_mindie_function_call(params, {"enable_auto_tool_choice": False})
        self.assertNotIn("mindie_tool_call_parser", params)
        self.assertNotIn("mindie_model_type", params)
        self.assertNotIn("mindie_chat_template", params)

    # FC 关闭 + 字段本就不存在 → 不抛 KeyError
    def test_fc_disabled_no_fields_no_error(self):
        params = {}
        _set_mindie_function_call(params, {"enable_auto_tool_choice": False})
        self.assertEqual(params, {})

    # FC 未传递（默认 False-y）→ 清除字段
    def test_fc_not_passed_removes_fields(self):
        params = {
            "mindie_tool_call_parser": "deepseek_v31",
            "mindie_model_type": "deepseekv2",
        }
        _set_mindie_function_call(params, {})
        self.assertNotIn("mindie_tool_call_parser", params)
        self.assertNotIn("mindie_model_type", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
