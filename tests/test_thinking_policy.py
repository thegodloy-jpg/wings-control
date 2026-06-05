# -*- coding: utf-8 -*-
"""enable_reasoning 关闭时「强制非思考」策略测试。

覆盖：
  - utils.model_utils.resolve_thinking_off_policy（模型名 → 关闭思考 kwargs / always_on / None）
  - proxy.thinking_policy.apply_to_chat_body（请求体改写 / 客户端覆盖压制 / always_on 仅告警）
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils.model_utils import resolve_thinking_off_policy, THINKING_ALWAYS_ON  # noqa: E402
import proxy.thinking_policy as tp  # noqa: E402


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


class TestApplyToChatBody(unittest.TestCase):
    """proxy 请求体改写。"""

    CHAT = "/v1/chat/completions"

    def _apply(self, env_val, body, path=None):
        with patch.dict(os.environ, ({"WINGS_THINKING_OFF": env_val} if env_val is not None else {}),
                        clear=False):
            if env_val is None:
                os.environ.pop("WINGS_THINKING_OFF", None)
            tp.reload_policy()
            return tp.apply_to_chat_body(json.dumps(body).encode(), path or self.CHAT)

    def test_dict_policy_injects_kwargs(self):
        out = self._apply('{"enable_thinking": false}', {"model": "Qwen3-32B", "messages": []})
        payload = json.loads(out)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_dict_policy_overrides_client_attempt(self):
        # 客户端试图开启 thinking，必须被强制压制为 false。
        out = self._apply('{"enable_thinking": false}',
                          {"messages": [], "chat_template_kwargs": {"enable_thinking": True}})
        payload = json.loads(out)
        self.assertIs(payload["chat_template_kwargs"]["enable_thinking"], False)

    def test_dict_policy_preserves_other_kwargs(self):
        out = self._apply('{"thinking": false}',
                          {"messages": [], "chat_template_kwargs": {"foo": 1}})
        payload = json.loads(out)
        self.assertEqual(payload["chat_template_kwargs"], {"foo": 1, "thinking": False})

    def test_always_on_does_not_modify_body(self):
        body = {"model": "DeepSeek-R1", "messages": []}
        out = self._apply("always_on", body)
        self.assertEqual(json.loads(out), body)

    def test_no_policy_is_noop(self):
        body = {"model": "Qwen3-32B", "messages": []}
        out = self._apply(None, body)
        self.assertEqual(json.loads(out), body)

    def test_non_chat_path_is_noop(self):
        body = {"prompt": "hi", "chat_template_kwargs": {"enable_thinking": True}}
        out = self._apply('{"enable_thinking": false}', body, path="/v1/completions")
        self.assertEqual(json.loads(out), body)

    def test_invalid_json_body_passthrough(self):
        with patch.dict(os.environ, {"WINGS_THINKING_OFF": '{"enable_thinking": false}'}):
            tp.reload_policy()
            raw = b"not-json"
            self.assertEqual(tp.apply_to_chat_body(raw, self.CHAT), raw)

    def test_invalid_env_value_disables_policy(self):
        out = self._apply("not-json-not-always_on", {"messages": []})
        self.assertNotIn("chat_template_kwargs", json.loads(out))

    def tearDown(self):
        os.environ.pop("WINGS_THINKING_OFF", None)
        tp.reload_policy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
