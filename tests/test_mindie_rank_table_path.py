# -*- coding: utf-8 -*-
"""MindIE rank table 路径校验单测。"""

import os
import sys
import tempfile
import unittest
import importlib
import json
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

    def test_copy_rank_table_normalizes_device_ids_per_server(self):
        rank_table = {
            "server_count": "2",
            "version": "1.0",
            "server_list": [
                {
                    "server_id": "112.254.176.102",
                    "device": [
                        {"device_ip": "10.20.1.3", "rank_id": "0", "device_id": "1"},
                        {"device_ip": "10.20.1.9", "rank_id": "1", "device_id": "7"},
                    ],
                    "container_ip": "112.254.176.102",
                },
                {
                    "server_id": "112.254.176.103",
                    "device": [
                        {"device_ip": "10.20.1.2", "rank_id": "2", "device_id": "0"},
                        {"device_ip": "10.20.1.5", "rank_id": "3", "device_id": "3"},
                    ],
                    "container_ip": "112.254.176.103",
                },
            ],
            "status": "completed",
        }
        src = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with src:
            json.dump(rank_table, src)
        self.addCleanup(lambda: os.path.exists(src.name) and os.remove(src.name))

        dst = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        dst.close()
        self.addCleanup(lambda: os.path.exists(dst.name) and os.remove(dst.name))

        mindie_adapter._copy_and_normalize_rank_table(src.name, dst.name)

        with open(dst.name, "r", encoding="utf-8") as f:
            normalized = json.load(f)
        server_devices = [server["device"] for server in normalized["server_list"]]
        self.assertEqual([device["device_id"] for device in server_devices[0]], ["0", "1"])
        self.assertEqual([device["device_id"] for device in server_devices[1]], ["0", "1"])
        self.assertEqual([device["device_ip"] for device in server_devices[0]], ["10.20.1.3", "10.20.1.9"])
        self.assertEqual([device["rank_id"] for device in server_devices[1]], ["2", "3"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
