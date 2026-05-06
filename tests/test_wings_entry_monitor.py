# -*- coding: utf-8 -*-
"""wings_entry monitor shell snippet tests."""
# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.wings_entry import _build_monitor_script  # noqa: E402


class TestWingsEntryMonitor(unittest.TestCase):
    def test_retry_progress_uses_default_script_start_epoch(self):
        script = _build_monitor_script(
            retry_cmd="echo retry &\nENGINE_PID=$!",
            engine="vllm_ascend",
        )

        self.assertIn('SCRIPT_START_EPOCH="${SCRIPT_START_EPOCH:-$(date +%s)}"', script)
        self.assertIn('START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")', script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
