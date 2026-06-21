#!/usr/bin/env python3
"""多 pod 分布式 DP 的 rpc-port / dp-address 一致性兜底回归。

复现根因：平台对每个 D pod 动态分配 VLLM_LLMDD_RPC_PORT（ephemeral 段）时，
wings 旧逻辑原样透传 → 4 个 pod 端口不一致 → vLLM DP 握手 5 分钟超时
（"Did not receive response from front-end process within 5 minutes"）。

兜底后：跨 pod（dp_size_local<dp_size）时 ephemeral 端口被强制改为角色固定常量，
dp-address 被对齐到 NODE_IPS[0]，保证同角色全 pod 派生一致。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wings_control"))

from core import config_loader as C  # noqa: E402

# 复现用户真实拓扑：D = 单节点 4 pod，DP8×TP1，local=2（跨 pod）
D_PODS = ["10.254.233.67", "10.254.224.229", "10.254.224.230", "10.254.224.231"]
DYN = ["41677", "53929", "47811", "39205"]  # 平台按 pod 动态分配的 ephemeral 端口


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
    return {
        "PD_ROLE": "D", "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "DP_SIZE_LOCAL": local,
        "Master_IP": master or D_PODS[0],
        "NODE_IPS": ",".join(D_PODS),
        "RANK_IP": D_PODS[idx],
        "VLLM_LLMDD_RPC_PORT": rpc,
    }


def test_multipod_ephemeral_rpc_forced_consistent():
    """4 个 pod 各拿不同 ephemeral 端口 → 兜底后全部归一到角色固定常量 12777。"""
    ports = {_derive(_env_for(i, DYN[i]))["rpc_port"] for i in range(4)}
    assert ports == {"12777"}, ports


def test_multipod_rank_and_address_consistent():
    """rank_start 仍按 NODE_IPS 位置派生；dp_address 全 pod 一致 = NODE_IPS[0]。"""
    rows = [_derive(_env_for(i, DYN[i])) for i in range(4)]
    assert [r["dp_rank_start"] for r in rows] == [0, 2, 4, 6], rows
    assert {r["dp_address"] for r in rows} == {"10.254.233.67"}, rows


def test_multipod_fixed_port_honored():
    """非 ephemeral 的固定端口（平台对全角色一致下发）不被覆盖。"""
    ports = {_derive(_env_for(i, "12321"))["rpc_port"] for i in range(4)}
    assert ports == {"12321"}, ports


def test_multipod_address_realigned_to_head():
    """dp-address 被误设为非 NODE_IPS[0]（如泄漏的某 P pod IP）→ 强制对齐到 head。"""
    got = _derive(_env_for(2, DYN[2], master="9.9.9.9"))
    assert got["dp_address"] == "10.254.233.67", got


def test_singlepod_ephemeral_not_touched():
    """单 pod（local==dp，全 rank 本地）ephemeral 端口仅本地用，不干预。"""
    env = {
        "PD_ROLE": "D", "PD_DECODE_DP_SIZE": "8", "PD_DECODE_TP_SIZE": "1",
        "PD_PREFILL_DP_SIZE": "2", "PD_PREFILL_TP_SIZE": "1",
        "DP_SIZE_LOCAL": "8", "Master_IP": "10.254.233.67",
        "NODE_IPS": "10.254.233.67", "RANK_IP": "10.254.233.67",
        "VLLM_LLMDD_RPC_PORT": "41677",
    }
    got = _derive(env)
    assert got["rpc_port"] == "41677", got
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
