#!/usr/bin/env python3
"""多 pod 分布式 DP 的 rpc-port / dp-address 处理契约回归。

契约（2026-06 变更）：--data-parallel-rpc-port 按角色【硬编码】（P=12890 / D=12777），
config_loader 刻意【不读】VLLM_LLMDD_RPC_PORT / PD_DP_RPC_PORT。理由：同角色每个 pod
各自算出同一常量 → DP 域 rpc-port 跨 pod 天然一致，无需平台协调。
部署侧须放行这两个固定端口（而非平台动态分配的 ephemeral 口）。

注：--data-parallel-address 仍【信任并原样透传】平台下发的 Master_IP（指向 DP 域 rank0），
不在 wings 侧改写——它常与 NetworkPolicy/端口放行绑定。
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


def test_rpc_port_hardcoded_ignores_ephemeral_env():
    """平台下发 ephemeral 端口 → 被忽略，D 恒为固定常量 12777。"""
    ports = {_derive(_env_for(i, "46982"))["rpc_port"] for i in range(4)}
    assert ports == {"12777"}, ports  # env 不参与，硬编码常量


def test_rpc_port_hardcoded_ignores_inconsistent_env():
    """平台误下发不一致端口 → 全部忽略，统一为硬编码常量（一致性由常量保证）。"""
    dyn = ["41677", "53929", "47811", "39205"]
    ports = {_derive(_env_for(i, dyn[i]))["rpc_port"] for i in range(4)}
    assert ports == {"12777"}, ports


def test_rpc_port_hardcoded_ignores_fixed_env():
    """即使平台下发固定端口也忽略，统一用 wings 硬编码常量。"""
    ports = {_derive(_env_for(i, "12321"))["rpc_port"] for i in range(4)}
    assert ports == {"12777"}, ports


def test_rpc_port_hardcoded_when_env_missing():
    """未下发任何 rpc env → D 恒为 12777。"""
    got = _derive(_env_for(0, None))
    assert got["rpc_port"] == "12777", got


def test_rpc_port_role_prefill_is_distinct():
    """P 角色硬编码为 12890，与 D（12777）不同 → 同机 P/D 不 bind 冲突。"""
    env = {
        "PD_ROLE": "P", "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "DP_SIZE_LOCAL": "2", "Master_IP": "10.254.13.83",
        "NODE_IPS": "10.254.13.83,10.254.13.67", "RANK_IP": "10.254.13.83",
        "VLLM_LLMDD_RPC_PORT": "46982",
    }
    got = _derive(env)
    assert got["rpc_port"] == "12890", got


def test_multipod_rank_consistent():
    """rank_start 按 NODE_IPS 位置派生（用 RANK_IP，非 HOST_IP）。"""
    rows = [_derive(_env_for(i, "46982")) for i in range(4)]
    assert [r["dp_rank_start"] for r in rows] == [0, 2, 4, 6], rows


def test_multipod_address_trusted_not_realigned():
    """dp_address 信任平台 Master_IP，wings 不强制对齐到 NODE_IPS[0]。"""
    got = _derive(_env_for(2, "46982", master="9.9.9.9"))
    assert got["dp_address"] == "9.9.9.9", got  # 原样透传，不改写


def test_singlepod_rpc_port_hardcoded():
    """单 pod（local==dp）同样忽略 env，D 恒为 12777。"""
    env = {
        "PD_ROLE": "D", "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "DP_SIZE_LOCAL": "8", "Master_IP": "10.254.13.83",
        "NODE_IPS": "10.254.13.83", "RANK_IP": "10.254.13.83",
        "VLLM_LLMDD_RPC_PORT": "46982",
    }
    got = _derive(env)
    assert got["rpc_port"] == "12777", got
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
