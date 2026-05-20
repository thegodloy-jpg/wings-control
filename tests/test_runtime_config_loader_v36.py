# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6.1 Phase D runtime config carrier tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.runtime_config_loader import (  # noqa: E402
    load_distributed_runtime_config,
    mindie_service_template_path,
)
from engines.mindie_adapter import _build_config_merge_script  # noqa: E402


class TestV36RuntimeConfigLoader(unittest.TestCase):
    def test_load_distributed_config_from_phase_d_carrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            distributed_dir = root / "distributed"
            distributed_dir.mkdir(parents=True)
            (distributed_dir / "defaults.yaml").write_text(
                json.dumps(
                    {
                        "master": {"port": 16123},
                        "workers": {"port": 16124},
                        "vllm_distributed": {"rpc_port": 14444},
                    }
                ),
                encoding="utf-8",
            )

            config = load_distributed_runtime_config(config_root=root)

        self.assertEqual(config["master"]["host"], "127.0.0.1")
        self.assertEqual(config["master"]["port"], 16123)
        self.assertEqual(config["workers"]["port"], 16124)
        self.assertEqual(config["vllm_distributed"]["ray_head_port"], 28020)
        self.assertEqual(config["vllm_distributed"]["rpc_port"], 14444)
        self.assertEqual(config["mindie_distributed"]["master_port"], 27070)

    def test_load_distributed_config_falls_back_to_builtin_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_distributed_runtime_config(config_root=Path(tmp))

        self.assertEqual(config["master"]["port"], 16000)
        self.assertEqual(config["workers"]["port"], 16001)
        self.assertEqual(config["sglang_distributed"]["dist_port"], 28030)

    def test_mindie_template_path_points_to_phase_d_carrier(self):
        path = mindie_service_template_path()

        self.assertEqual(
            path,
            "/opt/wings-control/config/templates/mindie_service_config.json",
        )

    def test_mindie_merge_script_uses_only_phase_d_template(self):
        script = _build_config_merge_script("{}", "/tmp/config.json", "/tmp")

        self.assertIn("/opt/wings-control/config/templates/mindie_service_config.json", script)
        self.assertNotIn("/opt/wings-control/config/defaults/mindie_service_config.json", script)
        self.assertNotIn("LEGACY_TEMPLATE", script)
        self.assertIn("template_path = LOCAL_TEMPLATE", script)


if __name__ == "__main__":
    unittest.main()
