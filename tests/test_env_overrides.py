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


if __name__ == "__main__":
    unittest.main(verbosity=2)
