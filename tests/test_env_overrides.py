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

from core.wings_entry import _build_env_overrides_preamble  # noqa: E402
from config.settings import settings  # noqa: E402


class TestEnvOverridesPreamble(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
