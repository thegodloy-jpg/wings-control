# -*- coding: utf-8 -*-
"""MindIE rank table 路径校验单测。"""

import os
import sys
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

mindie_adapter = importlib.import_module("engines.mindie_adapter")


class TestMindieRankTablePath(unittest.TestCase):
    def test_explicit_rank_table_path_must_exist(self):
        env = {"RANK_TABLE_PATH": "/missing/explicit_rank_table.json"}
        with patch.dict(os.environ, env, clear=False):
            with patch("engines.mindie_adapter.DEFAULT_RANK_TABLE_PATH", "/missing/default_rank_table.json"):
                with self.assertRaisesRegex(ValueError, "rank table file to exist"):
                    mindie_adapter._resolve_external_rank_table_path()

    def test_existing_explicit_rank_table_path_is_accepted(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with tmp:
            tmp.write("{}")
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))

        env = {"RANK_TABLE_PATH": tmp.name}
        with patch.dict(os.environ, env, clear=False):
            path = mindie_adapter._resolve_external_rank_table_path()

        self.assertEqual(path, tmp.name)

    def test_default_rank_table_path_must_exist_when_env_missing(self):
        env = {"RANK_TABLE_PATH": ""}
        with patch.dict(os.environ, env, clear=False):
            with patch("engines.mindie_adapter.DEFAULT_RANK_TABLE_PATH", "/missing/default_rank_table.json"):
                with self.assertRaisesRegex(ValueError, "/missing/default_rank_table.json"):
                    mindie_adapter._resolve_external_rank_table_path()

    def test_existing_default_rank_table_path_is_accepted(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with tmp:
            tmp.write("{}")
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))

        env = {"RANK_TABLE_PATH": ""}
        with patch.dict(os.environ, env, clear=False):
            with patch("engines.mindie_adapter.DEFAULT_RANK_TABLE_PATH", tmp.name):
                path = mindie_adapter._resolve_external_rank_table_path()

        self.assertEqual(path, tmp.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
