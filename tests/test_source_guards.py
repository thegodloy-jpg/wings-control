# -*- coding: utf-8 -*-
"""外部 env 脚本 source 的 nounset 守卫单测。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from engines import mindie_adapter, sglang_adapter, vllm_adapter  # noqa: E402


class TestSourceGuards(unittest.TestCase):
    def test_sglang_env_script_is_guarded_from_nounset(self):
        with patch("engines.sglang_adapter.os.path.exists", return_value=True):
            cmds = sglang_adapter._build_base_env_commands({}, str(ROOT))

        self.assertEqual(cmds[0], "set +u")
        self.assertIn("wings_source_env_with_diff", cmds[1])
        self.assertIn("set_sglang_env.sh", cmds[1])
        self.assertIn("else source", cmds[1])
        self.assertEqual(cmds[2], "set -u")

    def test_mindie_config_env_script_is_inlined(self):
        cmds = mindie_adapter._build_env_commands({})

        rendered = "\n".join(cmds)
        self.assertIn("# MindIE 单机引擎环境初始化脚本", rendered)
        self.assertIn("set +u", cmds)
        self.assertIn("set -u", cmds)
        self.assertIn("export NPU_MEMORY_FRACTION=0.96", cmds)
        self.assertIn(
            "wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh "
            "mindie/set_env.sh --backend=atb",
            rendered,
        )
        self.assertIn(
            "source /usr/local/Ascend/mindie/set_env.sh --backend=atb",
            rendered,
        )
        self.assertNotIn(
            f"source {mindie_adapter.root_dir}\\wings\\config\\set_mindie_single_env.sh",
            cmds,
        )

    def test_mindie_fallback_env_source_passes_backend_argument(self):
        cmds = mindie_adapter._build_ascend_env_source_commands()

        rendered = "\n".join(cmds)
        self.assertIn(
            "wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh "
            "mindie/set_env.sh --backend=atb",
            rendered,
        )
        self.assertIn(
            "source /usr/local/Ascend/mindie/set_env.sh --backend=atb",
            rendered,
        )

    def test_vllm_ascend_accel_patch_key_is_not_rewritten_to_vllm(self):
        source = (ROOT / "wings_control" / "core" / "wings_entry.py").read_text(encoding="utf-8")

        self.assertIn('"vllm_ascend": "vllm_ascend"', source)
        self.assertNotIn('"vllm_ascend": "vllm"', source)

    def test_v4_pro_static_engine_defaults_stay_out_of_vllm_adapter(self):
        source = (ROOT / "wings_control" / "engines" / "vllm_adapter.py").read_text(encoding="utf-8")

        self.assertNotIn("_DEEPSEEK_V4_PRO_CAPACITY_DEFAULTS", source)
        self.assertNotIn("_DEEPSEEK_V4_PRO_RUNTIME_DEFAULTS", source)
        self.assertNotIn("_DEEPSEEK_V4_PRO_ADDITIONAL_CONFIG", source)
        self.assertNotIn("engine_config[\"tensor_parallel_size\"] = 16", source)
        self.assertNotIn("engine_config[\"data_parallel_size\"] = 2", source)
        self.assertNotIn("engine_config[\"data_parallel_size_local\"] = 1", source)

    def test_glm47_w8a8_fingerprint_defaults_do_not_carry_speculative_config(self):
        adapter_source = (ROOT / "wings_control" / "engines" / "vllm_adapter.py").read_text(encoding="utf-8")
        ascend_defaults = (
            ROOT / "wings_control" / "config" / "defaults" / "ascend_default.json"
        ).read_text(encoding="utf-8")

        defaults_block = adapter_source.split("_GLM47_W8A8_ENGINE_DEFAULTS", 1)[1].split(
            "_GLM47_W8A8_DEEP_MERGE_KEYS", 1
        )[0]
        self.assertNotIn("speculative_config", defaults_block)
        self.assertNotIn("glm4_moe_mtp", ascend_defaults)


if __name__ == "__main__":
    unittest.main(verbosity=2)
