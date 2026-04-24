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

        self.assertEqual(cmds, ["set +u", f"source {ROOT}\\wings\\config\\set_sglang_env.sh", "set -u"])

    def test_mindie_dev_env_script_is_guarded_from_nounset(self):
        with patch("engines.mindie_adapter.os.path.exists", return_value=True):
            cmds = mindie_adapter._build_env_commands({})

        self.assertEqual(cmds[:3], [
            "set +u",
            f"source {mindie_adapter.root_dir}\\wings\\config\\set_mindie_single_env.sh",
            "set -u",
        ])

    def test_vllm_qwen3next_bisheng_source_is_guarded_from_nounset(self):
        fake_model = type("FakeModel", (), {"model_architecture": "Qwen3NextForCausalLM"})()
        with patch("engines.vllm_adapter.ModelIdentifier", return_value=fake_model):
            cmds = vllm_adapter._build_vllm_ascend_extensions({
                "model_name": "x",
                "model_path": "/tmp/model",
                "model_type": "auto",
            })

        self.assertEqual(cmds, [
            "set +u",
            "source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh",
            "set -u",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
