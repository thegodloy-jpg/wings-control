#!/usr/bin/env python3
"""PD KV 传输超时软默认回归。

PD 跨 pod 拉 KV（mooncake/ADXL ascend_direct_transport）在慢链路/冷启动时，引擎默认
~10000ms(10s) 易触发 connect timeout + status 103902（vllm-ascend#2970）。wings 在 PD 角色
env 注入 ASCEND_CONNECT_TIMEOUT / ASCEND_TRANSFER_TIMEOUT 的【软默认 120000ms，可被平台覆盖】，
不改注册表、非 PD / 非 ascend 不注入。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

import engines.vllm_adapter as va  # noqa: E402


def _pd_env(engine="vllm_ascend", role="D"):
    os.environ.pop("PD_ROLE", None)
    if role:
        os.environ["PD_ROLE"] = role
    try:
        return va._build_pd_role_env_commands(engine, "1.2.3.4", "eth0")
    finally:
        os.environ.pop("PD_ROLE", None)


def test_pd_ascend_emits_timeout_soft_defaults():
    cmds = _pd_env("vllm_ascend", "D")
    assert any(c == "export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-120000}" for c in cmds), cmds
    assert any(c == "export ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT:-120000}" for c in cmds), cmds


def test_overridable_uses_shell_default_pattern():
    """用 ${VAR:-默认} 形式 → 平台在 engine 容器设了值时其值优先（运行时解析）。"""
    cmds = _pd_env("vllm_ascend", "P")
    line = next(c for c in cmds if c.startswith("export ASCEND_CONNECT_TIMEOUT="))
    assert ":-120000}" in line and line.startswith("export ASCEND_CONNECT_TIMEOUT=${")


def test_non_pd_no_timeout():
    cmds = _pd_env("vllm_ascend", role=None)
    assert cmds == [], cmds


def test_nvidia_pd_no_ascend_timeout():
    cmds = _pd_env("vllm", "D")
    assert not any("ASCEND_CONNECT_TIMEOUT" in c for c in cmds), cmds


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
