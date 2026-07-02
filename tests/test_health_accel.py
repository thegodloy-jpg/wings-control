# -*- coding: utf-8 -*-
"""Unit tests for /v1/startup/accel data assembly in proxy/health_service.

回归三个问题（需求一 §补充）：
  1. 路径互串：advanced_features.json（使能+变体真相源）被正确读取，
     不再误读 log_analyzer 的安装状态 JSONL。
  2. /v1/startup/accel 的 data 保持 advanced_features.json 的
     engine/features/variants 结构，不再转换成特性数组。
并覆盖文件缺失 / 损坏时优雅降级为空 features/variants、不抛异常。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from proxy import health_service as hs  # noqa: E402


def _write(directory: Path, name: str, text: str) -> str:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestReadAdvancedFeaturesState(unittest.TestCase):
    def test_reads_engine_features_variants(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "advanced_features.json", json.dumps({
                "engine": "vllm_ascend",
                "features": {"speculative_decode": True, "sparse_kv": False},
                "variants": {"speculative_decode": "deepseek_mtp"},
            }))
            engine, features, variants = hs._read_advanced_features_state(p)
        self.assertEqual(engine, "vllm_ascend")
        self.assertTrue(features["speculative_decode"])
        self.assertEqual(variants["speculative_decode"], "deepseek_mtp")

    def test_missing_file_returns_empty(self):
        engine, features, variants = hs._read_advanced_features_state("/no/such/file.json")
        self.assertEqual((engine, features, variants), ("", {}, {}))

    def test_corrupt_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "advanced_features.json", "{not valid json")
            self.assertEqual(hs._read_advanced_features_state(p), ("", {}, {}))

    def test_non_dict_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "advanced_features.json", "[1, 2, 3]")
            self.assertEqual(hs._read_advanced_features_state(p), ("", {}, {}))


class TestBuildAccelData(unittest.TestCase):
    def test_engine_fallback(self):
        data = hs._build_accel_data("", {}, {})
        self.assertTrue(data["engine"])  # 回退到 ENGINE 环境变量默认 vllm
        self.assertEqual(data["features"], {})
        self.assertEqual(data["variants"], {})

    def test_keeps_file_shape(self):
        features = {"speculative_decode": True, "sparse_kv": False}
        variants = {"speculative_decode": "suffix", "sparse_kv": None}
        data = hs._build_accel_data("vllm_ascend", features, variants)
        self.assertEqual(data["engine"], "vllm_ascend")
        self.assertEqual(data["features"], features)
        self.assertEqual(data["variants"], variants)
        self.assertNotIn("version", data)


class TestStartupAccelEndpoint(unittest.TestCase):
    def test_returns_advanced_features_json_shape(self):
        payload = {
            "engine": "vllm_ascend",
            "features": {
                "speculative_decode": True,
                "sparse_kv": False,
                "kv_offload": True,
                "rag_acc": False,
            },
            "variants": {
                "speculative_decode": "suffix",
                "sparse_kv": None,
                "kv_offload": "lmcache_cpu",
                "rag_acc": None,
            },
        }
        old_path = hs.settings.ADVANCED_FEATURES_FILE
        try:
            with tempfile.TemporaryDirectory() as d:
                p = _write(Path(d), "advanced_features.json", json.dumps(payload))
                hs.settings.ADVANCED_FEATURES_FILE = p
                hs.app.state.client = object()

                response = TestClient(hs.app).get("/v1/startup/accel")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"code": 200, "msg": "", "data": payload})
        finally:
            hs.settings.ADVANCED_FEATURES_FILE = old_path


if __name__ == "__main__":
    unittest.main(verbosity=2)
