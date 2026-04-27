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

    def test_existing_export_echo_is_not_duplicated(self):
        script = (
            "export OMP_NUM_THREADS=10\n"
            'echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"\n'
            "export HCCL_CONNECT_TIMEOUT=7200\n"
            "printf '[mindie-env] HCCL_CONNECT_TIMEOUT=%s\\n' \"${HCCL_CONNECT_TIMEOUT:-}\"\n"
        )

        rendered = _inject_env_echo(script)

        self.assertEqual(rendered.count("[wings-env] export OMP_NUM_THREADS="), 1)
        self.assertEqual(rendered.count("[mindie-env] HCCL_CONNECT_TIMEOUT="), 1)
        self.assertNotIn("[wings-env] export HCCL_CONNECT_TIMEOUT=", rendered)

    def test_existing_command_echo_is_not_duplicated(self):
        script = (
            "echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server'\n"
            "exec python3 -m vllm.entrypoints.openai.api_server\n"
        )

        rendered = _inject_env_echo(script)

        self.assertEqual(rendered.count("[wings-cmd] >>>"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
