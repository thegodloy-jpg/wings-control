# -*- coding: utf-8 -*-
"""GLM-4.7-W8A8 引擎参数注入函数单测。"""

import os
import sys
import json
import unittest
from pathlib import Path

# 让测试可独立运行：把 wings_control 目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # wings-control/
sys.path.insert(0, str(ROOT / "wings_control"))

from engines.vllm_adapter import (  # noqa: E402
    _GLM47_W8A8_ENGINE_DEFAULTS,
    _deep_merge_user_priority,
    _is_w8a8_quantize,
    _inject_glm47_w8a8_engine_config,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _params(fixture_dir: str, engine: str = "vllm_ascend", engine_config=None):
    return {
        "engine": engine,
        "model_name": "test-model",
        "model_path": str(FIXTURES / fixture_dir),
        "model_type": "auto",
        "engine_config": dict(engine_config or {}),
    }


class TestIsW8A8Quantize(unittest.TestCase):
    def test_alias_hits(self):
        for q in ["w8a8", "W8A8", "smoothquant", "smooth_quant",
                  "ascend_w8a8", "ascend-w8a8", "w8a8_int8"]:
            self.assertTrue(_is_w8a8_quantize(q), q)

    def test_substring_hit(self):
        self.assertTrue(_is_w8a8_quantize("ascend-w8a8-int8"))
        self.assertTrue(_is_w8a8_quantize("xxx_W8A8_yyy"))

    def test_negative(self):
        for q in [None, "", "  ", "fp8", "bfloat16", "awq", "gptq", "int4"]:
            self.assertFalse(_is_w8a8_quantize(q), repr(q))


class TestDeepMerge(unittest.TestCase):
    def test_user_priority_scalar(self):
        self.assertEqual(_deep_merge_user_priority("u", "d"), "u")

    def test_default_fills_missing(self):
        self.assertEqual(
            _deep_merge_user_priority({"a": 1}, {"a": 2, "b": 3}),
            {"a": 1, "b": 3},
        )

    def test_recursive(self):
        u = {"x": {"y": 1}}
        d = {"x": {"y": 9, "z": 2}, "w": 5}
        self.assertEqual(
            _deep_merge_user_priority(u, d),
            {"x": {"y": 1, "z": 2}, "w": 5},
        )

    def test_none_treated_as_missing(self):
        self.assertEqual(
            _deep_merge_user_priority({"a": None}, {"a": 7}),
            {"a": 7},
        )


class TestInjectGlm47W8A8(unittest.TestCase):
    def test_w8a8_injects_all_defaults(self):
        p = _params("glm47_w8a8")
        _inject_glm47_w8a8_engine_config(p)
        for k, v in _GLM47_W8A8_ENGINE_DEFAULTS.items():
            self.assertEqual(p["engine_config"][k], v)
        self.assertNotIn("speculative_config", p["engine_config"])

    def test_user_value_not_overridden(self):
        p = _params("glm47_w8a8", engine_config={
            "enable_expert_parallel": False,
            "quantization": "user_custom",
        })
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"]["enable_expert_parallel"], False)
        self.assertEqual(p["engine_config"]["quantization"], "user_custom")
        self.assertEqual(p["engine_config"]["async_scheduling"], True)

    def test_additional_config_deep_merge(self):
        # 用户已有 additional_config（含一个不在注入集合内的自定义子键）→ 用户子键保留；注入子键追加
        p = _params("glm47_w8a8", engine_config={
            "additional_config": {
                "expert_tensor_parallel_size": 4,  # 用户自定义，注入函数不会动
            }
        })
        _inject_glm47_w8a8_engine_config(p)
        ac = p["engine_config"]["additional_config"]
        # 用户子键保留
        self.assertEqual(ac["expert_tensor_parallel_size"], 4)
        # 注入子键追加
        self.assertEqual(ac["enable_shared_expert_dp"], True)
        self.assertEqual(ac["ascend_fusion_config"], {"fusion_ops_gmmswigluquant": False})
        # 不应自动注入 ascend_scheduler_config（已从注入默认中移除）
        self.assertNotIn("ascend_scheduler_config", ac)

    def test_speculative_config_user_method_preserved(self):
        """显式 engine_config.speculative_config 是上层入口，GLM 指纹注入不得改写。"""
        p = _params("glm47_w8a8", engine_config={
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}
        })
        _inject_glm47_w8a8_engine_config(p)
        sc = p["engine_config"]["speculative_config"]
        self.assertEqual(sc["method"], "mtp")  # 用户优先
        self.assertEqual(sc["num_speculative_tokens"], 3)

    def test_json_string_speculative_config_is_preserved_without_fingerprint_merge(self):
        """架构指纹注入不是 spec 入口：JSON 字符串不应被解析后补默认字段。"""
        user_spec = json.dumps({"method": "custom_mtp"})
        p = _params("glm47_w8a8", engine_config={
            "speculative_config": user_spec
        })
        _inject_glm47_w8a8_engine_config(p)

        self.assertEqual(p["engine_config"]["speculative_config"], user_spec)

    def test_unparseable_string_dict_config_is_not_overridden(self):
        p = _params("glm47_w8a8", engine_config={
            "speculative_config": "user-owned-value"
        })
        _inject_glm47_w8a8_engine_config(p)

        self.assertEqual(p["engine_config"]["speculative_config"], "user-owned-value")

    def test_compilation_config_injected_when_absent(self):
        p = _params("glm47_w8a8")
        _inject_glm47_w8a8_engine_config(p)
        cc = p["engine_config"]["compilation_config"]
        self.assertEqual(cc["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(cc["cudagraph_capture_sizes"][0], 1)
        self.assertEqual(cc["cudagraph_capture_sizes"][-1], 128)

    def test_bf16_glm47_not_injected(self):
        p = _params("glm47_bf16")
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"], {})

    def test_glm45_same_arch_bf16_not_injected(self):
        p = _params("glm45_bf16")
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"], {})

    def test_non_vllm_engine_skipped(self):
        p = _params("glm47_w8a8", engine="mindie")
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"], {})

    def test_missing_model_path_safe(self):
        p = {
            "engine": "vllm_ascend",
            "model_name": "x",
            "model_path": "",
            "model_type": "auto",
            "engine_config": {},
        }
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"], {})

    def test_invalid_path_safe(self):
        p = _params("__nonexistent__")
        _inject_glm47_w8a8_engine_config(p)
        self.assertEqual(p["engine_config"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
