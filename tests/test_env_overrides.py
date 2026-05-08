# -*- coding: utf-8 -*-
"""env_overrides 注入逻辑单测。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from core.wings_entry import (  # noqa: E402
    _build_env_echo_helpers_preamble,
    _build_env_overrides_preamble,
)
from config.settings import settings  # noqa: E402


class TestEnvOverridesPreamble(unittest.TestCase):
    def test_source_env_helper_forwards_extra_arguments(self):
        preamble = _build_env_echo_helpers_preamble()

        self.assertIn("source \"$script_path\" \"$@\"", preamble)

    def test_shell_override_is_guarded_from_nounset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_dir = Path(tmpdir)
            (env_dir / "custom.sh").write_text(
                "export HCCL_BUFFSIZE=$HCCL_BUFFSIZE\n",
                encoding="utf-8",
            )

            with patch.object(settings, "ENV_OVERRIDES_DIR", str(env_dir)):
                preamble = _build_env_overrides_preamble()

        self.assertIn("set +u\nif command -v wings_source_env_with_diff", preamble)
        self.assertIn("wings_source_env_with_diff", preamble)
        self.assertIn("else source ", preamble)
        self.assertIn("\nset -u\n", preamble)
        self.assertIn("custom.sh", preamble)


class TestCollectEarsPatchFeatures(unittest.TestCase):
    """_collect_ears_patch_features 补丁安装层单测。

    验收标准：
    - vllm_ascend：不管投机推理以何种特性组合开启，始终返回 [] (不安装 EARS 补丁)
    - vllm：开启投机推理（无草稿模型）时返回 ["ears"]
    """

    def setUp(self):
        # 延迟导入，避免在模块加载时触发配置解析
        from core.wings_entry import _collect_ears_patch_features  # noqa: E402
        self._fn = _collect_ears_patch_features

    def _merged(self, engine="vllm_ascend", enable_spec=True, model_path=None):
        return {
            "engine": engine,
            "enable_speculative_decode": enable_spec,
            "speculative_decode_model_path": model_path or "",
            "model_name": "DeepSeek-V3.1-w8a8",
            "model_path": "/models/",
            "model_type": "",
        }

    def test_vllm_ascend_speculative_returns_empty(self):
        """vllm_ascend 单特性（仅投机推理）→ 不安装 EARS 补丁。"""
        result = self._fn("vllm_ascend", self._merged())
        self.assertEqual(result, [])

    def test_vllm_ascend_speculative_disabled_returns_empty(self):
        """vllm_ascend 未开启投机推理 → 不安装 EARS 补丁。"""
        result = self._fn("vllm_ascend", self._merged(enable_spec=False))
        self.assertEqual(result, [])

    def test_vllm_ascend_with_draft_model_returns_empty(self):
        """vllm_ascend 使用草稿模型 → 不安装 EARS 补丁。"""
        result = self._fn("vllm_ascend", self._merged(model_path="/models/draft/"))
        self.assertEqual(result, [])

    def test_vllm_engine_speculative_returns_ears(self):
        """vllm（非 Ascend）开启投机推理（无草稿模型）→ 返回 ['ears']。"""
        from unittest.mock import patch as _patch

        class _FakeDS:
            def __init__(self, *a, **kw):
                self.model_architecture = "DeepseekV3ForCausalLM"
                self.model_quantize = ""

        with _patch("engines.vllm_adapter.ModelIdentifier", _FakeDS):
            result = self._fn("vllm", self._merged(engine="vllm"))
        self.assertEqual(result, ["ears"])

    def test_vllm_engine_speculative_disabled_returns_empty(self):
        """vllm 未开启投机推理 → 不安装 EARS 补丁。"""
        result = self._fn("vllm", self._merged(engine="vllm", enable_spec=False))
        self.assertEqual(result, [])

    def test_non_vllm_engine_returns_empty(self):
        """非 vllm 引擎（如 mindie）→ 不安装 EARS 补丁。"""
        result = self._fn("mindie", self._merged(engine="mindie"))
        self.assertEqual(result, [])


class TestEarsFullAcceptanceCriteria(unittest.TestCase):
    """EARS 使能关闭（vllm_ascend）/ 开启（vllm）的完整三项验收测试。

    每个用例同时检查：
      1. 环境变量 ``VLLM_EARS_TOLERANCE=0.5`` 是否正确存在/缺失；
      2. 补丁安装命令层：``_collect_ears_patch_features`` 返回是否包含 ``ears``；
      3. 追加字段：``--speculative-config`` 是否正确追加到启动命令中。
    """

    def setUp(self):
        # 延迟导入
        from core.wings_entry import _collect_ears_patch_features
        from engines.vllm_adapter import build_start_script
        self._collect = _collect_ears_patch_features
        self._build = build_start_script

    def _make_params(self, engine: str) -> dict:
        return {
            "engine": engine,
            "model_name": "DeepSeek-V3",
            "model_path": "/models/ds",
            "model_type": "",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "",
            "engine_config": {"model": "/models/ds"},
        }

    def test_vllm_ascend_speculative_all_three_criteria_disabled(self):
        """vllm_ascend + 投机推理：三项均应体现「关闭」语义。"""
        class _FakeDS:
            def __init__(self, *a, **kw):
                self.model_architecture = "DeepseekV3ForCausalLM"
                self.model_quantize = ""

        params = self._make_params("vllm_ascend")
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDS):
            script = self._build(params)
        patch_features = self._collect("vllm_ascend", params)

        # 1. 环境变量：不注入
        self.assertNotIn("export VLLM_EARS_TOLERANCE=0.5", script,
                         "vllm_ascend 不应注入 VLLM_EARS_TOLERANCE")
        # 2. 补丁安装命令：EARS 补丁不被采集
        self.assertNotIn("ears", patch_features,
                         "vllm_ascend 不应采集 EARS 补丁")
        # 3. 追加字段：--speculative-config 正常生成（功能依然可用）
        self.assertIn("--speculative-config", script,
                      "vllm_ascend 投机推理的 --speculative-config 字段应正常追加")

    def test_vllm_speculative_all_three_criteria_enabled(self):
        """vllm（非 Ascend）+ 投机推理：三项均应体现「开启」语义。"""
        class _FakeDS:
            def __init__(self, *a, **kw):
                self.model_architecture = "DeepseekV3ForCausalLM"
                self.model_quantize = ""

        params = self._make_params("vllm")
        with patch("engines.vllm_adapter.ModelIdentifier", _FakeDS):
            script = self._build(params)
        patch_features = self._collect("vllm", params)

        # 1. 环境变量：注入
        self.assertIn("export VLLM_EARS_TOLERANCE=0.5", script,
                      "vllm 应注入 VLLM_EARS_TOLERANCE=0.5")
        # 2. 补丁安装命令：EARS 补丁被采集
        self.assertIn("ears", patch_features,
                      "vllm 应采集 EARS 补丁")
        # 3. 追加字段：--speculative-config 正常生成
        self.assertIn("--speculative-config", script,
                      "vllm 投机推理的 --speculative-config 字段应正常追加")


if __name__ == "__main__":
    unittest.main(verbosity=2)
