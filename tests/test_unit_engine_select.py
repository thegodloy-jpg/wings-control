# -*- coding: utf-8 -*-
"""UT-ES：引擎自动选择逻辑单元测试。

覆盖 TEST_PLAN.md GAP-4：
  - _select_nvidia_engine：5 条优先级分支
  - _select_ascend_engine：5 条优先级分支（含 Ascend310 强制 mindie）
  - _resolve_engine_choice：用户明确指定引擎 vs 自动选择

运行：
  python -m pytest tests/test_unit_engine_select.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
sys.path.insert(0, str(TESTS_DIR))

from core.config_loader import (  # noqa: E402
    _select_nvidia_engine,
    _select_ascend_engine,
    _resolve_engine_choice,
)

MOD = "core.config_loader"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(*, wings_supported=True, model_type="llm", arch="Qwen3ForCausalLM"):
    m = MagicMock()
    m.is_wings_supported.return_value = wings_supported
    m.identify_model_type.return_value = model_type
    m.model_architecture = arch
    return m


def _nvidia(gpu_usage_mode="full", wings_supported=True, model_type="llm",
            arch="Qwen3ForCausalLM", pd_role="", router=False):
    m = _make_model(wings_supported=wings_supported, model_type=model_type, arch=arch)
    with patch(f"{MOD}.get_pd_role_env", return_value=pd_role):
        with patch(f"{MOD}.get_router_env", return_value=router):
            return _select_nvidia_engine(gpu_usage_mode, m)


def _ascend(device_name="Ascend910B", wings_supported=True, model_type="llm",
            arch="Qwen3ForCausalLM", router=False):
    m = _make_model(wings_supported=wings_supported, model_type=model_type, arch=arch)
    with patch(f"{MOD}.get_router_env", return_value=router):
        return _select_ascend_engine(device_name, m)


# ---------------------------------------------------------------------------
# _select_nvidia_engine
# ---------------------------------------------------------------------------

class TestSelectNvidiaEngine(unittest.TestCase):
    """NVIDIA GPU 引擎自动选择逻辑（5 条优先级分支）。"""

    # 优先级1：PD 分离 → vllm
    def test_pd_role_forces_vllm(self):
        self.assertEqual(_nvidia(pd_role="P"), "vllm")

    # 优先级2：Wings Router → vllm
    def test_router_env_forces_vllm(self):
        self.assertEqual(_nvidia(router=True), "vllm")

    # 优先级3：MIG 模式 → vllm
    def test_mig_mode_forces_vllm(self):
        self.assertEqual(_nvidia(gpu_usage_mode="mig", wings_supported=True), "vllm")

    # 优先级4：embedding 模型 → vllm
    def test_embedding_model_type_forces_vllm(self):
        self.assertEqual(_nvidia(model_type="embedding"), "vllm")

    # 优先级4：rerank 模型 → vllm
    def test_rerank_model_type_forces_vllm(self):
        self.assertEqual(_nvidia(model_type="rerank"), "vllm")

    # 优先级5：wings 已验证模型 → sglang
    def test_wings_supported_model_returns_sglang(self):
        self.assertEqual(_nvidia(wings_supported=True), "sglang")

    # 兜底：未验证架构 → vllm
    def test_unknown_arch_not_wings_supported_returns_vllm(self):
        self.assertEqual(_nvidia(wings_supported=False, arch="UnknownArchForCausalLM"), "vllm")


# ---------------------------------------------------------------------------
# _select_ascend_engine
# ---------------------------------------------------------------------------

class TestSelectAscendEngine(unittest.TestCase):
    """Ascend NPU 引擎自动选择逻辑（5 条优先级分支）。"""

    # 优先级1：Ascend310 → mindie（强制）
    def test_ascend310_forces_mindie(self):
        self.assertEqual(_ascend(device_name="Ascend310P"), "mindie")

    # 优先级1：Ascend310 + embedding → ValueError
    def test_ascend310_embedding_raises_value_error(self):
        m = _make_model(model_type="embedding")
        with self.assertRaises(ValueError):
            with patch(f"{MOD}.get_router_env", return_value=False):
                _select_ascend_engine("Ascend310P", m)

    # 优先级2：embedding → vllm_ascend
    def test_embedding_model_returns_vllm_ascend(self):
        self.assertEqual(_ascend(model_type="embedding"), "vllm_ascend")

    # 优先级2：rerank → vllm_ascend
    def test_rerank_model_returns_vllm_ascend(self):
        self.assertEqual(_ascend(model_type="rerank"), "vllm_ascend")

    # 优先级3：Wings Router 开启 → vllm_ascend
    def test_router_env_returns_vllm_ascend(self):
        self.assertEqual(_ascend(router=True), "vllm_ascend")

    # 优先级4a：特殊架构列表成员 → vllm_ascend
    def test_qwen3next_in_special_arch_list_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="Qwen3NextForCausalLM"), "vllm_ascend")

    def test_deepseekv32_in_special_arch_list_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="DeepseekV32ForCausalLM"), "vllm_ascend")

    def test_kimik25_in_special_arch_list_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="KimiK25ForConditionalGeneration"), "vllm_ascend")

    def test_minimax_m2_in_special_arch_list_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="MiniMaxM2ForCausalLM"), "vllm_ascend")

    def test_glm4moe_in_special_arch_list_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="Glm4MoeForCausalLM"), "vllm_ascend")

    # 优先级6b：wings 已验证普通架构 → mindie（Ascend 上推荐引擎）
    def test_wings_supported_common_arch_returns_mindie(self):
        self.assertEqual(_ascend(arch="Qwen3ForCausalLM", wings_supported=True), "mindie")

    def test_deepseekv3_wings_supported_returns_mindie(self):
        """DeepseekV3ForCausalLM 不在特殊列表 → wings_supported → mindie。"""
        self.assertEqual(_ascend(arch="DeepseekV3ForCausalLM", wings_supported=True), "mindie")

    # 兜底：未验证架构 → vllm_ascend
    def test_unknown_arch_not_wings_supported_returns_vllm_ascend(self):
        self.assertEqual(_ascend(arch="UnknownArchForCausalLM", wings_supported=False),
                         "vllm_ascend")


# ---------------------------------------------------------------------------
# _resolve_engine_choice
# ---------------------------------------------------------------------------

class TestResolveEngineChoice(unittest.TestCase):
    """用户指定引擎 vs 自动选择的优先级门控。"""

    def _resolve(self, engine_in_params, device_type="nvidia",
                 device_name="NVIDIA H800 80G", gpu_usage_mode="full",
                 arch="Qwen3ForCausalLM", wings_supported=True):
        m = _make_model(wings_supported=wings_supported, arch=arch)
        cmd_params = {"gpu_usage_mode": gpu_usage_mode}
        if engine_in_params:
            cmd_params["engine"] = engine_in_params
        with patch(f"{MOD}.get_pd_role_env", return_value=""):
            with patch(f"{MOD}.get_router_env", return_value=False):
                return _resolve_engine_choice(
                    device_type, device_name, gpu_usage_mode, cmd_params, m
                )

    def test_user_specified_vllm_is_honored(self):
        """用户明确指定 vllm → 直接返回 vllm（最高优先级）。"""
        self.assertEqual(self._resolve("vllm"), "vllm")

    def test_user_specified_sglang_is_honored(self):
        """用户明确指定 sglang → 直接返回 sglang。"""
        self.assertEqual(self._resolve("sglang"), "sglang")

    def test_user_specified_vllm_ascend_is_honored(self):
        """用户明确指定 vllm_ascend → 直接返回 vllm_ascend。"""
        self.assertEqual(self._resolve("vllm_ascend"), "vllm_ascend")

    def test_user_specified_mindie_is_honored(self):
        """用户明确指定 mindie → 直接返回 mindie。"""
        self.assertEqual(self._resolve("mindie"), "mindie")

    def test_user_specified_invalid_engine_raises(self):
        """不合法引擎名 → ValueError。"""
        with self.assertRaises(ValueError):
            self._resolve("unknown_engine")

    def test_no_engine_nvidia_wings_supported_selects_sglang(self):
        """未指定引擎 + NVIDIA + wings 已验证 → 自动选择 sglang。"""
        self.assertEqual(self._resolve(None, device_type="nvidia", wings_supported=True),
                         "sglang")

    def test_no_engine_nvidia_unknown_arch_selects_vllm(self):
        """未指定引擎 + NVIDIA + 未验证架构 → 自动兜底 vllm。"""
        self.assertEqual(self._resolve(None, device_type="nvidia",
                                       arch="UnknownArchForCausalLM", wings_supported=False),
                         "vllm")

    def test_no_engine_ascend_common_arch_selects_mindie(self):
        """未指定引擎 + Ascend + 普通 wings 已验证架构 → 自动选择 mindie。"""
        self.assertEqual(self._resolve(None, device_type="ascend",
                                       device_name="Ascend910B", arch="Qwen3ForCausalLM",
                                       wings_supported=True),
                         "mindie")

    def test_no_engine_unknown_device_type_returns_vllm(self):
        """未指定引擎 + 未知设备类型 → 自动兜底 vllm。"""
        self.assertEqual(self._resolve(None, device_type="custom_accel"), "vllm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
