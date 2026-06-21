#!/usr/bin/env python3
"""多 pod 分布式 DP 的 rpc-port / dp-address 处理契约回归。

背景：多 pod（D 跨 pod）external-lb DP 要求 --data-parallel-rpc-port / --data-parallel-address
全 pod 一致。该一致性由【上层平台】负责（页面统一下发同一值，且常与 NetworkPolicy/端口放行绑定）。

契约：wings 一律【信任并原样透传】平台下发的 VLLM_LLMDD_RPC_PORT / Master_IP，
不在 wings 侧强改——强改（例如把 ephemeral 端口改成固定常量）会顶掉平台协调好的端口、
反而打断已放行的连通性。仅当未下发时由下游回退角色固定端口；ephemeral 值只打告警不改写。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

from core import config_loader as C  # noqa: E402

# 真实拓扑：D = 4 pod，DP8×TP1，local=2（跨 pod）
D_PODS = ["10.254.13.83", "10.254.13.67", "10.254.13.72", "10.254.13.85"]


def _clear():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP",
                "VLLM_LLMDD_RPC_PORT", "RANK_IP", "POD_IP"):
            os.environ.pop(k, None)


def _derive(env):
    _clear()
    os.environ.update(env)
    try:
        return C._get_pd_external_lb_params()
    finally:
        _clear()


def _env_for(idx, rpc, master=None, local="2"):
    e = {
        "PD_ROLE": "D", "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "DP_SIZE_LOCAL": local,
        "Master_IP": master or D_PODS[0],
        "NODE_IPS": ",".join(D_PODS),
        "RANK_IP": D_PODS[idx],
    }
    if rpc is not None:
        e["VLLM_LLMDD_RPC_PORT"] = rpc
    return e


def test_multipod_consistent_ephemeral_preserved():
    """平台对 4 pod 下发【相同】的 ephemeral 端口（真实修复后场景）→ 原样保留，不改写。"""
    ports = {_derive(_env_for(i, "46982"))["rpc_port"] for i in range(4)}
    assert ports == {"46982"}, ports  # 信任平台，不被改成 12777


def test_multipod_inconsistent_is_passthrough_not_masked():
    """平台误下发不一致端口 → wings 原样透传（不静默替换）；一致性须由平台保证。"""
    dyn = ["41677", "53929", "47811", "39205"]
    ports = [_derive(_env_for(i, dyn[i]))["rpc_port"] for i in range(4)]
    assert ports == dyn, ports  # 各自原样，wings 不强行归一（避免顶掉网络策略）


def test_multipod_fixed_port_preserved():
    """非 ephemeral 固定端口原样保留。"""
    ports = {_derive(_env_for(i, "12321"))["rpc_port"] for i in range(4)}
    assert ports == {"12321"}, ports


def test_multipod_rank_consistent():
    """rank_start 按 NODE_IPS 位置派生（用 RANK_IP，非 HOST_IP）。"""
    rows = [_derive(_env_for(i, "46982")) for i in range(4)]
    assert [r["dp_rank_start"] for r in rows] == [0, 2, 4, 6], rows


def test_multipod_address_trusted_not_realigned():
    """dp_address 信任平台 Master_IP，wings 不强制对齐到 NODE_IPS[0]。"""
    got = _derive(_env_for(2, "46982", master="9.9.9.9"))
    assert got["dp_address"] == "9.9.9.9", got  # 原样透传，不改写


def test_missing_rpc_port_left_empty_for_downstream_fallback():
    """未下发 VLLM_LLMDD_RPC_PORT → config_loader 返回空，交由 vllm_adapter 回退角色固定端口。"""
    got = _derive(_env_for(0, None))
    assert got["rpc_port"] == "", got


def test_singlepod_ephemeral_preserved():
    """单 pod（local==dp）同样信任 env，原样保留。"""
    env = {
        "PD_ROLE": "D", "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "DP_SIZE_LOCAL": "8", "Master_IP": "10.254.13.83",
        "NODE_IPS": "10.254.13.83", "RANK_IP": "10.254.13.83",
        "VLLM_LLMDD_RPC_PORT": "46982",
    }
    got = _derive(env)
    assert got["rpc_port"] == "46982", got
    assert got["dp_rank_start"] == 0, got


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
