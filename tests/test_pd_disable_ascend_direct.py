#!/usr/bin/env python3
"""部署级可选规避 PD_DISABLE_ASCEND_DIRECT 的回归。

GLM5 注册表 extra_config 默认 use_ascend_direct=true（官方 ADXL 直传）。某些部署（如 1P1D 测试）
的 ADXL/RDMA 数据面不通会报 ascend_direct_transport 连 P 超时 + status 103902（vllm-ascend#2970）。
本开关让【单个部署】通过 env 关掉 ADXL（不改共享注册表 / 不影响其它部署）。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

from core import config_loader as C  # noqa: E402

_ENTRY = json.load(open(os.path.join(ROOT, "wings_control", "config", "defaults", "pd_config.json"),
                        encoding="utf-8"))["pd_config"]["GlmMoeDsaForCausalLM"]
_EXT = {"role": "P", "dp_size": 1, "tp_size": 16}


def _extra(disable):
    for k in ("PD_DISABLE_ASCEND_DIRECT",):
        os.environ.pop(k, None)
    os.environ["PD_DECODE_DP_SIZE"] = "1"
    os.environ["PD_DECODE_TP_SIZE"] = "16"
    if disable is not None:
        os.environ["PD_DISABLE_ASCEND_DIRECT"] = disable
    try:
        return C._build_pd_external_lb_kv(_ENTRY, dict(_EXT))["kv_connector_extra_config"]
    finally:
        os.environ.pop("PD_DISABLE_ASCEND_DIRECT", None)


def test_default_keeps_official_adxl():
    """不设 env → 保留官方 use_ascend_direct（注册表不变，其它部署/4机不受影响）。"""
    assert _extra(None).get("use_ascend_direct") is True


def test_env_disables_adxl():
    """设 PD_DISABLE_ASCEND_DIRECT=1 → 从 kv extra 移除 use_ascend_direct。"""
    assert "use_ascend_direct" not in _extra("1")


def test_env_truthy_variants():
    for v in ("1", "true", "TRUE", "yes", "on"):
        assert "use_ascend_direct" not in _extra(v), v


def test_env_falsy_keeps_adxl():
    for v in ("0", "false", "", "no"):
        assert _extra(v).get("use_ascend_direct") is True, v


def test_topology_preserved_after_disable():
    """关掉 ADXL 不影响 prefill/decode 拓扑字段。"""
    extra = _extra("1")
    assert extra["prefill"] == {"dp_size": 1, "tp_size": 16}
    assert extra["decode"] == {"dp_size": 1, "tp_size": 16}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
