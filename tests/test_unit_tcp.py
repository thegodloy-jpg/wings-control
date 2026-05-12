# -*- coding: utf-8 -*-
"""UT-TCP parser-mapping 测试：验证各架构×引擎组合下 tool_call_parser / reasoning_parser 正确注入。

覆盖 TEST_PLAN.md 中 UT-TCP-01~24 共 24 个用例。

用例编号与方案完全对齐：
  UT-TCP-01~14  vLLM + NVIDIA (14 个架构/变体)
  UT-TCP-15~19  vLLM-Ascend + Ascend NPU (5 个)
  UT-TCP-20~24  SGLang + NVIDIA (5 个)

运行：
  python -m pytest tests/test_unit_tcp.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from core.config_loader import _set_function_call, load_and_merge_configs  # noqa: E402
from engines.mindie_adapter import _build_model_config_overrides  # noqa: E402
from snapshot_framework import (  # noqa: E402
    FakeModelIdentifier,
    make_launch_args,
    nvidia_hardware,
    ascend_hardware,
)

# ---------------------------------------------------------------------------
# Helper: 通过 load_and_merge_configs 获取 engine_config 子字典
# ---------------------------------------------------------------------------

def _load_engine_config(
    hardware: dict,
    architecture: str,
    engine: str,
    model_name: str = "test-model",
    model_path: str = "/models/test",
    enable_auto_tool_choice: bool = True,
    **overrides,
) -> dict:
    """运行完整配置合并流程，返回 engine_config 子字典。"""
    fake_model = FakeModelIdentifier(
        architecture=architecture,
        model_name=model_name,
        model_path=model_path,
    )

    def _make_fake(name, path, mtype):
        return fake_model

    known_args = make_launch_args(
        engine=engine,
        model_name=model_name,
        model_path=model_path,
        enable_auto_tool_choice=enable_auto_tool_choice,
        **overrides,
    ).to_namespace()

    with (
        patch("core.config_loader.ModelIdentifier", side_effect=_make_fake),
        patch("core.config_loader.get_local_ip", return_value="10.0.0.1"),
        patch("core.config_loader.check_permission_640", return_value=True),
    ):
        merged = load_and_merge_configs(hardware_env=hardware, known_args=known_args)
    return merged.get("engine_config", {})


# ---------------------------------------------------------------------------
# UT-TCP-01~14  vLLM + NVIDIA
# ---------------------------------------------------------------------------

class TestNvidiaVllmParserMapping(unittest.TestCase):
    """UT-TCP-01~14：NVIDIA + vLLM 架构/变体的 tool_call_parser / reasoning_parser 映射。"""

    HW = nvidia_hardware(device_count=8, memory_gb=80)

    def _ec(self, arch, model_name="test-model", **kw):
        return _load_engine_config(self.HW, arch, "vllm", model_name=model_name, **kw)

    # UT-TCP-01
    def test_tcp01_qwen3_hermes_qwen3(self):
        """Qwen3ForCausalLM → tool_call_parser=hermes / reasoning_parser=qwen3。"""
        ec = self._ec("Qwen3ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertEqual(ec.get("reasoning_parser"), "qwen3")

    # UT-TCP-02
    def test_tcp02_qwen3moe_default_hermes_qwen3(self):
        """Qwen3MoeForCausalLM（default）→ hermes / qwen3。"""
        ec = self._ec("Qwen3MoeForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertEqual(ec.get("reasoning_parser"), "qwen3")

    # UT-TCP-03
    def test_tcp03_qwen3moe_coder_480b_qwen3_xml(self):
        """Qwen3MoeForCausalLM + Qwen3-Coder-480B-A35B-Instruct → qwen3_xml（无 reasoning）。"""
        ec = self._ec("Qwen3MoeForCausalLM",
                      model_name="Qwen3-Coder-480B-A35B-Instruct")
        self.assertEqual(ec.get("tool_call_parser"), "qwen3_xml")
        self.assertNotIn("reasoning_parser", ec)

    # UT-TCP-04
    def test_tcp04_qwen3moe_coder_30b_qwen3_xml(self):
        """Qwen3MoeForCausalLM + Qwen3-Coder-30B-A3B-Instruct → qwen3_xml（无 reasoning）。"""
        ec = self._ec("Qwen3MoeForCausalLM",
                      model_name="Qwen3-Coder-30B-A3B-Instruct")
        self.assertEqual(ec.get("tool_call_parser"), "qwen3_xml")
        self.assertNotIn("reasoning_parser", ec)

    # UT-TCP-05
    def test_tcp05_deepseekv3_default_deepseek_v3_r1(self):
        """DeepseekV3ForCausalLM（default）→ deepseek_v3 / deepseek_r1。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3")
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v3")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_r1")

    # UT-TCP-06
    def test_tcp06_deepseekv3_v31_deepseek_v31(self):
        """DeepseekV3ForCausalLM + DeepSeek-V3.1 → deepseek_v31 / deepseek_v3。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3.1")
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v31")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_v3")

    # UT-TCP-07
    def test_tcp07_deepseekv32_deepseek_v32_r1(self):
        """DeepseekV32ForCausalLM → deepseek_v32 / deepseek_r1。"""
        ec = self._ec("DeepseekV32ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v32")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_r1")

    # UT-TCP-08
    def test_tcp08_glm4moe_glm47_glm45(self):
        """Glm4MoeForCausalLM → glm47 / glm45。"""
        ec = self._ec("Glm4MoeForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "glm47")
        self.assertEqual(ec.get("reasoning_parser"), "glm45")

    # UT-TCP-09
    def test_tcp09_glm_moe_dsa_glm47_glm45(self):
        """GlmMoeDsaForCausalLM → glm47 / glm45。"""
        ec = self._ec("GlmMoeDsaForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "glm47")
        self.assertEqual(ec.get("reasoning_parser"), "glm45")

    # UT-TCP-10
    def test_tcp10_llama_llama3_json_no_reasoning(self):
        """LlamaForCausalLM → llama3_json（无 reasoning_parser）。"""
        ec = self._ec("LlamaForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "llama3_json")
        self.assertNotIn("reasoning_parser", ec)

    # UT-TCP-11
    def test_tcp11_glm4_hermes_no_reasoning(self):
        """Glm4ForCausalLM → hermes（无 reasoning_parser）。"""
        ec = self._ec("Glm4ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertNotIn("reasoning_parser", ec)

    # UT-TCP-12
    def test_tcp12_qwen3next_hermes_qwen3(self):
        """Qwen3NextForCausalLM → hermes / qwen3。"""
        ec = self._ec("Qwen3NextForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertEqual(ec.get("reasoning_parser"), "qwen3")

    # UT-TCP-13
    def test_tcp13_minimax_m2_minimax_m2(self):
        """MiniMaxM2ForCausalLM → minimax_m2 / minimax_m2_append_think。"""
        ec = self._ec("MiniMaxM2ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "minimax_m2")
        self.assertEqual(ec.get("reasoning_parser"), "minimax_m2_append_think")

    # UT-TCP-14
    def test_tcp14_fc_disabled_no_parser(self):
        """enable_auto_tool_choice=False → 任何架构均无 tool_call_parser / reasoning_parser。"""
        ec = _load_engine_config(
            self.HW, "Qwen3ForCausalLM", "vllm",
            enable_auto_tool_choice=False,
        )
        self.assertNotIn("tool_call_parser", ec)
        self.assertNotIn("reasoning_parser", ec)


# ---------------------------------------------------------------------------
# UT-TCP-15~19  vLLM-Ascend + Ascend NPU
# ---------------------------------------------------------------------------

class TestAscendVllmAscendParserMapping(unittest.TestCase):
    """UT-TCP-15~19：Ascend NPU + vllm_ascend 架构的 parser 映射。"""

    HW = ascend_hardware(device_count=8)

    def _ec(self, arch, model_name="test-model"):
        return _load_engine_config(self.HW, arch, "vllm_ascend", model_name=model_name)

    # UT-TCP-15
    def test_tcp15_kimik25_kimi_k2(self):
        """KimiK25ForConditionalGeneration → kimi_k2 / kimi_k2。"""
        ec = self._ec("KimiK25ForConditionalGeneration")
        self.assertEqual(ec.get("tool_call_parser"), "kimi_k2")
        self.assertEqual(ec.get("reasoning_parser"), "kimi_k2")

    # UT-TCP-16
    def test_tcp16_deepseekv3_r1_w8a8_deepseek_v3(self):
        """DeepseekV3ForCausalLM + DeepSeek-R1-w8a8（Ascend）→ deepseek_v3 / deepseek_r1。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-R1-w8a8")
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v3")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_r1")

    # UT-TCP-17
    def test_tcp17_deepseekv3_v31_ascend_deepseek_v31(self):
        """DeepseekV3ForCausalLM + DeepSeek-V3.1（Ascend）→ deepseek_v31 / deepseek_v3。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3.1")
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v31")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_v3")

    def test_tcp17b_deepseekv31_w8a8_ascend_uses_v31_parser(self):
        """DeepseekV3ForCausalLM + DeepSeek-V3.1-w8a8（Ascend）→ 复用 V3.1 专属 parser。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3.1-w8a8")
        self.assertIs(ec.get("enable_auto_tool_choice"), True)
        self.assertEqual(ec.get("tool_call_parser"), "deepseek_v31")
        self.assertEqual(ec.get("reasoning_parser"), "deepseek_v3")

    def test_tcp17c_deepseekv31_w8a8_fc_disabled_no_parser(self):
        """DeepSeek-V3.1-w8a8 未开启 FC 时，不应注入 tool_call_parser / reasoning_parser。"""
        ec = _load_engine_config(
            self.HW,
            "DeepseekV3ForCausalLM",
            "vllm_ascend",
            model_name="DeepSeek-V3.1-w8a8",
            enable_auto_tool_choice=False,
        )

        self.assertNotIn("enable_auto_tool_choice", ec)
        self.assertNotIn("tool_call_parser", ec)
        self.assertNotIn("reasoning_parser", ec)

    # UT-TCP-18
    def test_tcp18_qwen3moe_ascend_hermes_qwen3(self):
        """Qwen3MoeForCausalLM（Ascend）→ hermes / qwen3。"""
        ec = self._ec("Qwen3MoeForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertEqual(ec.get("reasoning_parser"), "qwen3")

    # UT-TCP-19
    def test_tcp19_glm_moe_dsa_ascend_glm47_glm45(self):
        """GlmMoeDsaForCausalLM（Ascend）→ glm47 / glm45。"""
        ec = self._ec("GlmMoeDsaForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "glm47")
        self.assertEqual(ec.get("reasoning_parser"), "glm45")


# ---------------------------------------------------------------------------
# UT-TCP-20~24  SGLang + NVIDIA
# ---------------------------------------------------------------------------

class TestNvidiaSglangParserMapping(unittest.TestCase):
    """UT-TCP-20~24：NVIDIA + SGLang 架构的 tool_call_parser 映射。

    SGLang 不支持 reasoning_parser，全部断言其不存在。
    """

    HW = nvidia_hardware(device_count=8, memory_gb=80)

    def _ec(self, arch, model_name="test-model"):
        return _load_engine_config(self.HW, arch, "sglang", model_name=model_name)

    # UT-TCP-20
    def test_tcp20_deepseekv3_sglang_deepseekv3(self):
        """DeepseekV3ForCausalLM + sglang → deepseekv3（单词无下划线）。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3")
        self.assertEqual(ec.get("tool_call_parser"), "deepseekv3")

    def test_deepseekv31_sglang_h20_uses_deepseekv31(self):
        """DeepSeek-V3.1 + SGLang + H20 → deepseekv31。"""
        with patch.dict(os.environ, {"WINGS_H20_MODEL": "H20-96G"}):
            ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3.1")
        self.assertEqual(ec.get("tool_call_parser"), "deepseekv31")

    def test_deepseekv32_sglang_uses_deepseekv32(self):
        """DeepseekV32ForCausalLM + SGLang → deepseekv32。"""
        ec = self._ec("DeepseekV32ForCausalLM", model_name="DeepSeek-V3.2")
        self.assertEqual(ec.get("tool_call_parser"), "deepseekv32")

    def test_qwen3_coder_sglang_uses_qwen3_coder(self):
        """Qwen3-Coder + SGLang → qwen3_coder。"""
        ec = self._ec("Qwen3MoeForCausalLM", model_name="Qwen3-Coder-480B-A35B-Instruct")
        self.assertEqual(ec.get("tool_call_parser"), "qwen3_coder")

    # UT-TCP-21
    def test_tcp21_qwen3_sglang_qwen(self):
        """Qwen3ForCausalLM + sglang → qwen。"""
        ec = self._ec("Qwen3ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "qwen")

    # UT-TCP-22
    def test_tcp22_qwen3moe_sglang_qwen(self):
        """Qwen3MoeForCausalLM + sglang → qwen。"""
        ec = self._ec("Qwen3MoeForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "qwen")

    # UT-TCP-23
    def test_tcp23_llama_sglang_llama3(self):
        """LlamaForCausalLM + sglang → llama3。"""
        ec = self._ec("LlamaForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "llama3")

    # UT-TCP-24
    def test_tcp24_glm4moe_sglang_glm(self):
        """Glm4MoeForCausalLM + sglang → glm。"""
        ec = self._ec("Glm4MoeForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "glm")

    def test_qwen35_sglang_uses_qwen(self):
        """Qwen3.5 + SGLang → qwen。"""
        ec = self._ec("Qwen3_5ForConditionalGeneration")
        self.assertEqual(ec.get("tool_call_parser"), "qwen")

    def test_glm4_sglang_uses_glm(self):
        """Glm4ForCausalLM + SGLang → glm。"""
        ec = self._ec("Glm4ForCausalLM")
        self.assertEqual(ec.get("tool_call_parser"), "glm")

    def test_config_file_kebab_case_keys_override_sglang_defaults(self):
        """config-file 中 CLI 风格 kebab-case key 应归一化后覆盖 SGLang 默认值。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "sglang_kebab_config.json"
            config_path.write_text(
                json.dumps({
                    "context-length": 32768,
                    "mem-fraction-static": 0.95,
                    "max-running-requests": 128,
                    "tool-call-parser": "hermes",
                    "grammar-backend": "xgrammar",
                }),
                encoding="utf-8",
            )

            ec = _load_engine_config(
                self.HW,
                "DeepseekV32ForCausalLM",
                "sglang",
                model_name="DeepSeek-V3.2",
                config_file=str(config_path),
            )

        self.assertEqual(ec.get("context_length"), 32768)
        self.assertEqual(ec.get("mem_fraction_static"), 0.95)
        self.assertEqual(ec.get("max_running_requests"), 128)
        self.assertEqual(ec.get("tool_call_parser"), "hermes")
        self.assertEqual(ec.get("grammar_backend"), "xgrammar")
        self.assertNotIn("tool-call-parser", ec)


class TestAscendMindieParserMapping(unittest.TestCase):
    """Ascend NPU + MindIE 架构/变体的 Function Call 映射。"""

    HW = ascend_hardware(device_count=16)

    def _ec(self, arch, model_name="test-model", **kw):
        return _load_engine_config(self.HW, arch, "mindie", model_name=model_name, **kw)

    def test_deepseekv31_w8a8_mindie_injects_tool_call_options(self):
        """DeepSeek-V3.1-w8a8 单机 MindIE FC 应注入 models.deepseekv2.tool_call_options。"""
        ec = self._ec("DeepseekV3ForCausalLM", model_name="DeepSeek-V3.1-w8a8")

        self.assertEqual(ec.get("mindie_tool_call_parser"), "deepseek_v31")
        self.assertEqual(ec.get("mindie_model_type"), "deepseekv2")

        overrides = _build_model_config_overrides(ec, is_distributed=False, world_size=16)
        self.assertEqual(
            overrides["models"],
            {"deepseekv2": {"tool_call_options": {"tool_call_parser": "deepseek_v31"}}},
        )

    def test_deepseekv31_w8a8_mindie_fc_disabled_drops_internal_fields(self):
        """DeepSeek-V3.1-w8a8 未开启 FC 时，MindIE 内部 FC 字段不应残留。"""
        ec = self._ec(
            "DeepseekV3ForCausalLM",
            model_name="DeepSeek-V3.1-w8a8",
            enable_auto_tool_choice=False,
        )

        self.assertNotIn("mindie_tool_call_parser", ec)
        self.assertNotIn("mindie_model_type", ec)

        overrides = _build_model_config_overrides(ec, is_distributed=False, world_size=16)
        self.assertNotIn("models", overrides)

    def test_config_file_can_force_deepseekv31_mindie_function_call(self):
        """config-file 可强制设置 MindIE DeepSeek-V3.1 FC 内部字段并触发注入。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mindie_force_fc.json"
            config_path.write_text(
                json.dumps({
                    "mindie_model_type": "deepseekv2",
                    "mindie_tool_call_parser": "deepseek_v31",
                }),
                encoding="utf-8",
            )

            ec = self._ec(
                "DeepseekV3ForCausalLM",
                model_name="DeepSeek-V3.1-w8a8",
                enable_auto_tool_choice=False,
                config_file=str(config_path),
            )

        self.assertEqual(ec.get("mindie_tool_call_parser"), "deepseek_v31")
        self.assertEqual(ec.get("mindie_model_type"), "deepseekv2")

        overrides = _build_model_config_overrides(ec, is_distributed=False, world_size=16)
        self.assertEqual(
            overrides["models"],
            {"deepseekv2": {"tool_call_options": {"tool_call_parser": "deepseek_v31"}}},
        )


# ---------------------------------------------------------------------------
# 附加：_set_function_call 门控逻辑直接单元测试（方案外补充）
# ---------------------------------------------------------------------------

class TestSetFunctionCallGating(unittest.TestCase):
    """直接测试 _set_function_call 门控逻辑（与方案 UT-TCP 互补）。"""

    def test_fc_enabled_with_parser_keeps_both(self):
        params = {"tool_call_parser": "hermes", "reasoning_parser": "qwen3"}
        _set_function_call(params, {"enable_auto_tool_choice": True})
        self.assertEqual(params.get("tool_call_parser"), "hermes")
        self.assertEqual(params.get("reasoning_parser"), "qwen3")

    def test_fc_disabled_removes_all_parser_fields(self):
        params = {"tool_call_parser": "deepseek_v3", "reasoning_parser": "deepseek_r1"}
        _set_function_call(params, {"enable_auto_tool_choice": False})
        self.assertNotIn("tool_call_parser", params)
        self.assertNotIn("reasoning_parser", params)

    def test_fc_enabled_no_parser_removes_flag(self):
        params = {}
        _set_function_call(params, {"enable_auto_tool_choice": True})
        self.assertNotIn("tool_call_parser", params)
        self.assertNotIn("enable_auto_tool_choice", params)

    def test_fc_not_passed_removes_parsers(self):
        params = {"tool_call_parser": "hermes"}
        _set_function_call(params, {})
        self.assertNotIn("tool_call_parser", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
