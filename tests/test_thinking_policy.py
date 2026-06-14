# -*- coding: utf-8 -*-
"""enable_auto_think_choice 思考默认状态策略测试（对称开关）。

覆盖：
  - utils.model_utils.resolve_thinking_off_policy（模型名 → 思考 kwargs 键名 / always_on / None）
  - core.config_loader._set_thinking_default（生成端：启动命令注入服务级默认思考状态，
    开关开→强制思考、关→强制非思考；仅 vllm/vllm_ascend；请求体可覆盖、不兜底）
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils.model_utils import resolve_thinking_off_policy, THINKING_ALWAYS_ON  # noqa: E402
from core import config_loader as cl  # noqa: E402


class TestResolveThinkingOffPolicy(unittest.TestCase):
    """模型名 → 关闭思考策略解析。"""

    def test_qwen3_uses_enable_thinking(self):
        self.assertEqual(resolve_thinking_off_policy("Qwen3-32B"), {"enable_thinking": False})
        self.assertEqual(resolve_thinking_off_policy("Qwen3-235B-A22B"), {"enable_thinking": False})
        self.assertEqual(resolve_thinking_off_policy("Qwen3-Next-80B-A3B-Instruct"),
                         {"enable_thinking": False})

    def test_glm_moe_uses_enable_thinking(self):
        for name in ("GLM-4.5", "GLM-4.6", "GLM-4.7", "GLM-5", "GLM-5.1"):
            self.assertEqual(resolve_thinking_off_policy(name), {"enable_thinking": False}, name)

    def test_deepseek_v3_uses_thinking_key(self):
        # 注意：DeepSeek 用的键名是 thinking，不是 enable_thinking。
        self.assertEqual(resolve_thinking_off_policy("DeepSeek-V3.1"), {"thinking": False})
        self.assertEqual(resolve_thinking_off_policy("DeepSeek-V3.2-Exp"), {"thinking": False})
        self.assertEqual(resolve_thinking_off_policy("DeepSeek-V3"), {"thinking": False})

    def test_always_on_reasoners(self):
        for name in ("DeepSeek-R1", "DeepSeek-R1-0528", "DeepSeek-R1-Distill-Qwen-32B",
                     "QwQ-32B", "MiniMax-M2.5"):
            self.assertEqual(resolve_thinking_off_policy(name), THINKING_ALWAYS_ON, name)

    def test_non_thinking_models_return_none(self):
        for name in ("Qwen2.5-32B-Instruct", "LLaMA3.1-70B", "GLM-4-9B-0414", "bge-m3", ""):
            self.assertIsNone(resolve_thinking_off_policy(name), name)

    def test_always_on_takes_priority_over_family(self):
        # R1-Distill-Qwen 同时含 r1 与 qwen → 必须判为 always_on（R1 无法关闭思考）。
        self.assertEqual(resolve_thinking_off_policy("DeepSeek-R1-Distill-Qwen-7B"),
                         THINKING_ALWAYS_ON)


class _ModelInfoStub:
    def __init__(self, model_name: str):
        self.model_name = model_name


class TestSetThinkingDefault(unittest.TestCase):
    """生成端：启动命令注入 default_chat_template_kwargs（仅 vllm/vllm_ascend）。"""

    def _run(self, model_name, enable_auto_think_choice, params=None):
        params = params if params is not None else {}
        cl._set_thinking_default(
            params,
            {"enable_auto_think_choice": enable_auto_think_choice},
            _ModelInfoStub(model_name),
        )
        return params

    def test_qwen3_injects_enable_thinking_false(self):
        params = self._run("Qwen3-32B", False)
        self.assertEqual(params.get("default_chat_template_kwargs"), {"enable_thinking": False})

    def test_glm_moe_injects_enable_thinking_false(self):
        params = self._run("GLM-5.1", False)
        self.assertEqual(params.get("default_chat_template_kwargs"), {"enable_thinking": False})

    def test_deepseek_v3_injects_thinking_false(self):
        params = self._run("DeepSeek-V3.1", False)
        self.assertEqual(params.get("default_chat_template_kwargs"), {"thinking": False})

    def test_always_on_does_not_inject(self):
        params = self._run("DeepSeek-R1", False)
        self.assertNotIn("default_chat_template_kwargs", params)

    def test_non_thinking_does_not_inject(self):
        params = self._run("Qwen2.5-32B-Instruct", False)
        self.assertNotIn("default_chat_template_kwargs", params)

    def test_enabled_qwen3_injects_enable_thinking_true(self):
        # 开启推理 → 服务级默认【强制打开】思考。
        params = self._run("Qwen3-32B", True)
        self.assertEqual(params.get("default_chat_template_kwargs"), {"enable_thinking": True})

    def test_enabled_deepseek_v3_injects_thinking_true(self):
        params = self._run("DeepSeek-V3.1", True)
        self.assertEqual(params.get("default_chat_template_kwargs"), {"thinking": True})

    def test_enabled_overrides_residual_default_to_true(self):
        # 开启推理时把残留的非思考默认覆盖为思考。
        params = self._run("Qwen3-32B", True,
                           params={"default_chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(params.get("default_chat_template_kwargs"), {"enable_thinking": True})

    def test_enabled_always_on_does_not_inject(self):
        # 始终推理模型开启时无需注入（天生思考）。
        params = self._run("DeepSeek-R1", True)
        self.assertNotIn("default_chat_template_kwargs", params)

    def test_enabled_non_thinking_does_not_inject(self):
        # 非思考模型即便开启也无法强制思考。
        params = self._run("Qwen2.5-32B-Instruct", True)
        self.assertNotIn("default_chat_template_kwargs", params)


class TestThinkingSwitchUnsupportedEngineWarning(unittest.TestCase):
    """sglang / mindie 触发思考开关 → 日志提醒（启动期无法切换，行为不变）。"""

    def _capture(self, engine, enabled):
        import io
        import logging
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        cl.logger.addHandler(handler)
        old = cl.logger.level
        cl.logger.setLevel(logging.WARNING)
        try:
            cl._warn_thinking_switch_unsupported_engine(
                engine, {"enable_auto_think_choice": enabled})
        finally:
            cl.logger.removeHandler(handler)
            cl.logger.setLevel(old)
        return buf.getvalue()

    def test_sglang_on_warns(self):
        self.assertIn("不支持启动期思考开关", self._capture("sglang", True))

    def test_mindie_on_warns(self):
        self.assertIn("不支持启动期思考开关", self._capture("mindie", True))

    def test_off_does_not_warn(self):
        self.assertEqual(self._capture("sglang", False), "")
        self.assertEqual(self._capture("mindie", False), "")

    def test_vllm_engines_do_not_warn(self):
        self.assertEqual(self._capture("vllm", True), "")
        self.assertEqual(self._capture("vllm_ascend", True), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
