# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6 Phase A hardware canonical ID tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.hardware_detect import detect_hardware, normalize_chip_id  # noqa: E402


class TestV36HardwareDetect(unittest.TestCase):
    def test_normalize_chip_alias(self):
        chip, aliases = normalize_chip_id("910b-32g")
        self.assertEqual(chip, "910b-32")
        self.assertEqual(aliases[-1], "910b-32")

    def test_detect_chip_from_explicit_argument(self):
        with patch.dict(os.environ, {}, clear=True):
            result = detect_hardware(chip="rtx_pro_5000")
        self.assertEqual(result["chip_id"], "rtx-pro-5000")
        self.assertEqual(result["chip_source"], "cli")

    def test_detect_ascend_910b_32_from_hardware_file(self):
        payload = {
            "device": "ascend",
            "count": 1,
            "details": [{"name": "Ascend910B", "total_memory": 32}],
            "units": "GB",
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with patch.dict(os.environ, {"WINGS_HARDWARE_FILE": path}, clear=True):
                result = detect_hardware()
        finally:
            os.unlink(path)
        self.assertEqual(result["device"], "ascend")
        self.assertEqual(result["chip_id"], "910b-32")
        self.assertEqual(result["chip_source"], "inferred")

    def test_wings_chip_env_overrides_file(self):
        payload = {
            "device": "ascend",
            "count": 1,
            "chip_id": "910b-64",
            "details": [{"name": "Ascend910B", "total_memory": 64}],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with patch.dict(os.environ, {"WINGS_HARDWARE_FILE": path, "WINGS_CHIP": "910b-32g"}, clear=True):
                result = detect_hardware()
        finally:
            os.unlink(path)
        self.assertEqual(result["chip_id"], "910b-32")
        self.assertEqual(result["chip_source"], "cli")


if __name__ == "__main__":
    unittest.main()
