#!/usr/bin/env python3
"""方案 B 回归：PD 分离下在 env 汇总层中心剔除 VLLM_ASCEND_BALANCE_SCHEDULING。

vLLM-Ascend(≥0.20.2) 的 enable_balance_scheduling 与 PD 分离（kv_role=producer/consumer）
互斥，设置会 ValidationError 拒绝启动。各模型 env builder（GLM5/Kimi/MiniMax…）无条件注入
该 flag，wings 在 _build_vllm_common_env_cmds 汇总层按 PD 角色统一剔除（全模型 / 全启动路径）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

import engines.vllm_adapter as va  # noqa: E402

BS = "export VLLM_ASCEND_BALANCE_SCHEDULING=1"


def _run(pd_role, commands):
    os.environ.pop("PD_ROLE", None)
    if pd_role:
        os.environ["PD_ROLE"] = pd_role
    try:
        return va._filter_pd_incompatible_env(list(commands))
    finally:
        os.environ.pop("PD_ROLE", None)


def test_drop_under_pd_producer():
    out = _run("P", ["export FOO=1", BS, "export BAR=2"])
    assert out == ["export FOO=1", "export BAR=2"], out


def test_drop_under_pd_consumer():
    out = _run("D", [BS, "export X=1"])
    assert BS not in out, out


def test_keep_without_pd_role():
    out = _run(None, ["export FOO=1", BS])
    assert BS in out, out


def test_noop_when_flag_absent():
    out = _run("P", ["export FOO=1", "export BAR=2"])
    assert out == ["export FOO=1", "export BAR=2"], out


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
