# -*- coding: utf-8 -*-
"""_inject_env_echo 行为单测。"""

import sys
import unittest
from pathlib import Path

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import _inject_env_echo  # noqa: E402


class TestInjectEnvEcho(unittest.TestCase):
    def test_all_exported_variables_are_echoed(self):
        script = (
            "export LD_LIBRARY_PATH=/tmp/lib:${LD_LIBRARY_PATH:-}\n"
            "export OMP_NUM_THREADS=8\n"
            "export HCCL_BUFFSIZE=512\n"
        )

        rendered = _inject_env_echo(script)

        self.assertIn(
            'echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"\n',
            rendered,
        )
        self.assertIn(
            'echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"\n',
            rendered,
        )
        self.assertIn(
            'echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"\n',
            rendered,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
