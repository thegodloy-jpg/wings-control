# -*- coding: utf-8 -*-
"""分布式 start_command 分发与落盘行为单测。"""

import asyncio
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from config.settings import settings  # noqa: E402
from core.start_command_utils import write_start_command  # noqa: E402
from distributed.worker import EngineStartRequest, start_engine_api  # noqa: E402
from wings_control import DispatchOptions, _try_dispatch_to_worker  # noqa: E402


def _assert_start_command_mode(testcase: unittest.TestCase, script_path: Path) -> None:
    """Assert the start-command file keeps engine-container readable perms."""
    mode = stat.S_IMODE(script_path.stat().st_mode)
    if os.name == "nt":
        testcase.assertEqual(mode, 0o666)
    else:
        testcase.assertEqual(mode, 0o644)


class TestStartCommandWriter(unittest.TestCase):
    def test_write_start_command_is_world_readable_for_engine_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, "SHARED_VOLUME_PATH", tmpdir):
                script_path = Path(write_start_command("echo worker-ray\n"))
                self.assertTrue(script_path.exists())
                self.assertEqual(script_path.read_text(encoding="utf-8"), "echo worker-ray\n")
                _assert_start_command_mode(self, script_path)


class TestWorkerStartEngineApi(unittest.TestCase):
    def test_worker_prefers_prebuilt_start_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(settings, "SHARED_VOLUME_PATH", tmpdir):
                result = asyncio.run(start_engine_api(EngineStartRequest(
                    engine="vllm",
                    params={"distributed": True, "node_rank": 1},
                    start_command="echo prebuilt-ray-worker\n",
                )))
                script_path = Path(tmpdir) / settings.START_COMMAND_FILENAME

                self.assertEqual(result["status"], "started")
                self.assertEqual(
                    result["message"],
                    "Prebuilt engine start script written to shared volume",
                )
                self.assertEqual(
                    script_path.read_text(encoding="utf-8"),
                    "echo prebuilt-ray-worker\n",
                )
                _assert_start_command_mode(self, script_path)


class TestWorkerDispatch(unittest.TestCase):
    def test_dispatch_payload_contains_prebuilt_start_command(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "started"}

        with patch("wings_control.requests.post", return_value=response) as post:
            ok = _try_dispatch_to_worker(
                worker_ip="10.0.0.2",
                worker_port=15000,
                rank=1,
                params={"engine": "vllm", "distributed": True},
                options=DispatchOptions(max_retries=1),
                start_command="echo dispatched-ray-worker\n",
            )

        self.assertTrue(ok)
        self.assertEqual(
            post.call_args.kwargs["json"]["start_command"],
            "echo dispatched-ray-worker\n",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
