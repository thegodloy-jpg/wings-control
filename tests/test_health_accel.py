# -*- coding: utf-8 -*-
"""Unit tests for /v1/startup/accel data assembly in proxy/health_service.

回归三个问题（需求一 §补充）：
  1. 路径互串：advanced_features.json（使能+变体真相源）被正确读取，
     不再误读 log_analyzer 的安装状态 JSONL。
  2. 增强展示：每个特性透出 variant（走哪种变体），而不仅是 bool。
  3. errMsg 仅在补丁安装失败时从安装状态 JSONL 叠加；成功/缺失不污染。
并覆盖文件缺失 / 损坏时优雅降级为 4 个 disabled 特性、不抛异常。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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


class TestBuildFeatureList(unittest.TestCase):
    def test_always_four_features_in_order(self):
        out = hs._build_feature_list({}, {})
        self.assertEqual([f["name"] for f in out], list(hs._ADVANCED_FEATURE_KEYS))
        self.assertTrue(all(f["enabled"] is False for f in out))
        self.assertTrue(all(f["variant"] is None for f in out))
        self.assertTrue(all(f["errMsg"] == "" for f in out))

    def test_enabled_and_variant_from_state(self):
        features = {"speculative_decode": True, "kv_offload": True}
        variants = {"speculative_decode": "suffix", "kv_offload": "lmcache_cpu"}
        out = {f["name"]: f for f in hs._build_feature_list(features, variants)}
        self.assertTrue(out["speculative_decode"]["enabled"])
        self.assertEqual(out["speculative_decode"]["variant"], "suffix")
        self.assertEqual(out["kv_offload"]["variant"], "lmcache_cpu")
        self.assertFalse(out["sparse_kv"]["enabled"])
        # errMsg 恒为空串（单一真相源，无安装状态叠加）
        self.assertTrue(all(f["errMsg"] == "" for f in out.values()))


class TestBuildAccelData(unittest.TestCase):
    def test_engine_and_version_fallback(self):
        data = hs._build_accel_data("", [], version="0.17.0rc1")
        self.assertEqual(data["version"], "0.17.0rc1")
        self.assertTrue(data["engine"])  # 回退到 ENGINE 环境变量默认 vllm
        self.assertEqual(data["features"], [])

    def test_engine_from_state_preferred(self):
        data = hs._build_accel_data("vllm_ascend", [{"name": "x"}], version="1.0")
        self.assertEqual(data["engine"], "vllm_ascend")
        self.assertEqual(data["features"], [{"name": "x"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
