"""Sanity check for V4-Pro identifier/scope boundaries."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wings_control"))
import engines.vllm_adapter as va

cases = [
    ("DeepSeek-V4-Flash", "/m/DeepSeek-V4-Flash-w8a8-mtp", False, "Flash must not match Pro"),
    ("DeepSeek-V4-Flash-w8a8-mtp", "/m/DeepSeek-V4-Flash", False, "Flash exact name"),
    ("DeepSeek-V4-Pro-w4a8-mtp", "/m/DeepSeek-V4-Pro-w4a8-mtp", True, "Canonical Pro"),
    ("deepseek_v4_pro", "/m/deepseek_v4_pro", True, "underscore variant"),
    ("DeepSeekV4Pro", "/m/DeepSeekV4Pro", True, "concatenated"),
    ("DeepSeek-V3.1", "/m/DeepSeek-V3.1", False, "V3.1 unrelated"),
    ("DeepSeek-V3-w8a8", "/m/DeepSeek-V3-w8a8", False, "V3 baseline"),
    ("GLM-5.1", "/m/glm-5.1", False, "GLM-5.1 unrelated"),
    ("Qwen3-32B", "/m/Qwen3-32B", False, "Qwen unrelated"),
    ("Kimi-K2.5", "/m/Kimi-K2.5", False, "Kimi unrelated"),
    ("MiniMax-M2", "/m/MiniMax-M2", False, "MiniMax unrelated"),
    ("DeepSeek-V4-Pro-Flash-mix", "/m/x", False, "Both keywords → Flash wins (excludes Pro)"),
]
ok_count = 0
for name, path, expected, desc in cases:
    params = {"model_name": name, "model_path": path}
    actual = va._is_deepseek_v4_pro_params(params)
    status = "OK" if actual == expected else "FAIL"
    if actual == expected:
        ok_count += 1
    print(f"[{status}] {desc:50s} name={name!r:42s} -> {actual} (want {expected})")

print(f"\n{ok_count}/{len(cases)} identifier cases passed")

# scope cases: only A3 + distributed + nnodes=2 should be in scope
scope_cases = [
    # (params, expected, desc)
    ({"model_name": "DeepSeek-V4-Pro-w4a8-mtp", "engine": "vllm_ascend", "device_count": 16,
      "engine_config": {"served_model_name": "DeepSeek-V4-Pro-w4a8-mtp", "ascend_platform": "a3"},
      "distributed": True, "nnodes": 2}, True, "A3 dual-node: in scope"),
    ({"model_name": "DeepSeek-V4-Pro-w4a8-mtp", "engine": "vllm_ascend", "device_count": 8,
      "engine_config": {"served_model_name": "DeepSeek-V4-Pro-w4a8-mtp", "ascend_platform": "a3"},
      "distributed": False, "nnodes": 1}, False, "A2 single-node: out of scope"),
    ({"model_name": "DeepSeek-V4-Pro-w4a8-mtp", "engine": "vllm_ascend", "device_count": 16,
      "engine_config": {"served_model_name": "DeepSeek-V4-Pro-w4a8-mtp", "ascend_platform": "a3"},
      "distributed": False, "nnodes": 1}, False, "A3 single-node: out of scope"),
    ({"model_name": "DeepSeek-V4-Pro-w4a8-mtp", "engine": "vllm_ascend", "device_count": 32,
      "engine_config": {"served_model_name": "DeepSeek-V4-Pro-w4a8-mtp", "ascend_platform": "a3"},
      "distributed": True, "nnodes": 4}, False, "A3 4-node: out of scope"),
    ({"model_name": "DeepSeek-V4-Flash-w8a8-mtp", "engine": "vllm_ascend", "device_count": 16,
      "engine_config": {"served_model_name": "DeepSeek-V4-Flash-w8a8-mtp"},
      "distributed": True, "nnodes": 2}, False, "V4-Flash dual: out of scope"),
    ({"model_name": "DeepSeek-V4-Pro-w4a8-mtp", "engine": "vllm", "device_count": 16,
      "engine_config": {"served_model_name": "DeepSeek-V4-Pro-w4a8-mtp", "ascend_platform": "a3"},
      "distributed": True, "nnodes": 2}, False, "Non-Ascend: out of scope"),
]
ok2 = 0
for params, expected, desc in scope_cases:
    actual = va.is_deepseek_v4_pro_adapted_scope(params)
    status = "OK" if actual == expected else "FAIL"
    if actual == expected:
        ok2 += 1
    print(f"[{status}] {desc:50s} -> {actual} (want {expected})")
print(f"\n{ok2}/{len(scope_cases)} scope cases passed")
