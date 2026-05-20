# -*- coding: utf-8 -*-
# pyright: reportMissingImports=false
"""v3.6 Phase A CLI compatibility tests."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.start_args_compat import parse_launch_args  # noqa: E402


class TestV36LaunchArgs(unittest.TestCase):
    def _parse(self, *args: str):
        return parse_launch_args(["--model-name", "m", "--model-path", "/m", *args])

    def test_engine_alias_vllm_ascend_normalized(self):
        with patch.dict(os.environ, {}, clear=True):
            parsed = self._parse("--engine", "vllm-ascend")
        self.assertEqual(parsed.engine, "vllm_ascend")

    def test_chip_argument_is_preserved_and_exported(self):
        with patch.dict(os.environ, {}, clear=True):
            parsed = self._parse("--chip", "910b-32g")
            self.assertEqual(os.environ.get("WINGS_CHIP"), "910b-32g")
        self.assertEqual(parsed.chip, "910b-32g")

    def test_allow_experimental_from_env(self):
        with patch.dict(os.environ, {"WINGS_ALLOW_EXPERIMENTAL": "1"}, clear=True):
            parsed = self._parse()
        self.assertTrue(parsed.allow_experimental)
        self.assertFalse(parsed.no_experimental)

    def test_no_experimental_overrides_env_allow(self):
        with patch.dict(os.environ, {"WINGS_ALLOW_EXPERIMENTAL": "1"}, clear=True):
            parsed = self._parse("--no-experimental")
        self.assertFalse(parsed.allow_experimental)
        self.assertTrue(parsed.no_experimental)

    def test_allow_and_no_experimental_conflict(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                self._parse("--allow-experimental", "--no-experimental")

    def test_emit_resolved_params_argument_is_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            parsed = self._parse("--emit-resolved-params", "/tmp/resolved_params.json")
        self.assertEqual(parsed.emit_resolved_params, "/tmp/resolved_params.json")


if __name__ == "__main__":
    unittest.main()
