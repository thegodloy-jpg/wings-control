# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6.1 Phase D engine defaults loader tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.engine_defaults_loader import (  # noqa: E402
    engine_default_ref,
    load_engine_defaults,
    normalize_engine_default_id,
)
from core.config_loader import _load_default_config, _load_engine_fallback_defaults  # noqa: E402


class TestV36EngineDefaultsLoader(unittest.TestCase):
    def test_vllm_ascend_reuses_vllm_defaults(self):
        self.assertEqual(normalize_engine_default_id("vllm_ascend"), "vllm")
        defaults = load_engine_defaults("vllm_ascend")
        self.assertEqual(defaults["max_model_len"], 4096)
        self.assertEqual(defaults["gpu_memory_utilization"], 0.9)

    def test_load_engine_defaults_from_phase_d_carrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            carrier = root / "engine_defaults" / "sglang.yaml"
            carrier.parent.mkdir(parents=True)
            carrier.write_text(json.dumps({"context_length": 8192, "trust_remote_code": True}), encoding="utf-8")

            defaults = load_engine_defaults("sglang", config_root=root)
            ref = engine_default_ref("sglang", config_root=root)

        self.assertEqual(defaults["context_length"], 8192)
        self.assertTrue(ref.endswith("engine_defaults\\sglang.yaml") or ref.endswith("engine_defaults/sglang.yaml"))

    def test_config_loader_uses_phase_d_engine_defaults(self):
        with patch("core.config_loader.load_engine_defaults", return_value={"max_model_len": 1234}) as loader:
            defaults = _load_default_config({"device": "nvidia"})

        loader.assert_called_once_with("vllm")
        self.assertEqual(defaults["max_model_len"], 1234)

    def test_engine_fallback_uses_phase_d_engine_defaults(self):
        with patch("core.config_loader.load_engine_defaults", return_value={"context_length": 2048}) as loader:
            defaults = _load_engine_fallback_defaults("sglang")

        loader.assert_called_once_with("sglang")
        self.assertEqual(defaults["context_length"], 2048)


if __name__ == "__main__":
    unittest.main()
