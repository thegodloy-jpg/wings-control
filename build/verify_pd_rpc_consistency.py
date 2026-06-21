#!/usr/bin/env python3
"""验证：PD external-lb 多 pod 分布式 D，wings 对 --data-parallel-rpc-port /
--data-parallel-address 是否做一致性兜底。

复现用户真实拓扑：D = 单节点内 4 个 pod（各 2 卡），DP8×TP1，local=2。
rank 映射对齐其日志：head=10.254.233.67(idx0)，10.254.224.230(idx2)→rank_start=4(rank 4,5)。

两种模式：
  A) consistent —— 4 个 pod VLLM_LLMDD_RPC_PORT 同值（正确基线）
  B) dynamic    —— 平台按 pod 动态分配（ephemeral 段），4 个 pod 各不同（模拟故障）

结论看：wings 是否原样透传出不一致的 --data-parallel-rpc-port 而不报错。
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import dry_run  # 复用 create_mock_model_dir + 注入 wings_control 到 sys.path

from core.start_args_compat import parse_launch_args          # noqa: E402
from core.port_plan import derive_port_plan                   # noqa: E402
from core.wings_entry import build_launcher_plan              # noqa: E402
from config.settings import settings                          # noqa: E402

ARCH = "Qwen3MoeForCausalLM"
MODEL = "Qwen3-30B-A3B"

# D 的 4 个 pod（顺序即 NODE_IPS 顺序，决定 rank_start = idx * local）
D_PODS = ["10.254.233.67", "10.254.224.229", "10.254.224.230", "10.254.224.231"]
HEAD = D_PODS[0]
LOCAL = 2          # 每 pod fork 2 个 rank
TP = 1
DP = 8            # 4 pod × 2

# KV 全局拓扑（对齐日志 kv_connector_extra_config）
PF_DP, PF_TP = 2, 1
DC_DP, DC_TP = DP, TP

# 模式 B：模拟平台按 pod 动态分配（Linux ephemeral 32768-60999）
DYNAMIC_RPC = ["41677", "53929", "47811", "39205"]
CONSISTENT_RPC = "12777"


def _clean_env():
    for k in list(os.environ):
        if k.startswith(("PD_", "DP_", "TP_")) or k in (
                "NODE_IPS", "HOST_IP", "Master_IP", "MASTER_IP",
                "VLLM_LLMDD_RPC_PORT", "RANK_IP", "POD_IP"):
            os.environ.pop(k, None)


def gen_pod_cmd(host_ip: str, rpc: str) -> str:
    """按某个 D pod 的 env 生成 start_command，返回完整脚本。"""
    _clean_env()
    model_dir = dry_run.create_mock_model_dir(
        ARCH, {"quantization_config": {"quant_method": "ascend"}})
    try:
        os.environ.update({
            "WINGS_DEVICE": "ascend", "WINGS_ASCEND_PLATFORM": "a3",
            "DEVICE_COUNT": str(LOCAL * TP),
            "RANK_IP": host_ip,
            "PD_ROLE": "D",
            "DP_SIZE_LOCAL": str(LOCAL),
            "Master_IP": HEAD,                 # 4 个 pod 都指 head（一致）
            "VLLM_LLMDD_RPC_PORT": rpc,        # ← 唯一变量：模式 A 同值 / 模式 B 各异
            "NODE_IPS": ",".join(D_PODS),
            "PD_PREFILL_DP_SIZE": str(PF_DP), "PD_PREFILL_TP_SIZE": str(PF_TP),
            "PD_DECODE_DP_SIZE": str(DC_DP), "PD_DECODE_TP_SIZE": str(DC_TP),
        })
        la = parse_launch_args([
            "--model-name", MODEL, "--model-path", model_dir,
            "--engine", "vllm_ascend", "--device-count", str(LOCAL * TP),
            "--nnodes", "1", "--node-rank", "0"])
        pp = derive_port_plan(port=la.port,
                              enable_reason_proxy=settings.ENABLE_REASON_PROXY,
                              health_port=settings.HEALTH_PORT)
        return build_launcher_plan(la, pp).command
    finally:
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)


def extract(cmd: str) -> dict:
    rpc = re.search(r"--data-parallel-rpc-port\s+(\S+)", cmd)
    addr = re.search(r"--data-parallel-address\s+(?:'([^']*)'|(\S+))", cmd)
    start = re.search(r"RANK=\$\(\((\d+)\s*\+\s*i\)\)", cmd)
    local = re.search(r"for i in \$\(seq 0 (\d+)\)", cmd)
    addr_val = (addr.group(1) or addr.group(2)) if addr else "?"
    start_i = int(start.group(1)) if start else -1
    local_n = int(local.group(1)) + 1 if local else -1
    ranks = ",".join(str(start_i + i) for i in range(local_n)) if start_i >= 0 else "?"
    return {
        "rpc": rpc.group(1) if rpc else "?",
        "addr": addr_val,
        "ranks": ranks,
    }


def run_mode(name: str, rpc_for_idx) -> list:
    print(f"\n{'='*78}\n模式 {name}\n{'='*78}")
    print(f"{'pod (RANK_IP)':<20}{'--dp-rank':<12}{'--dp-address':<20}{'--dp-rpc-port':<14}")
    print("-" * 66)
    rows = []
    for idx, ip in enumerate(D_PODS):
        cmd = gen_pod_cmd(ip, rpc_for_idx(idx))
        info = extract(cmd)
        role = " (head/rank0)" if idx == 0 else ""
        print(f"{ip+role:<20}{info['ranks']:<12}{info['addr']:<20}{info['rpc']:<14}")
        rows.append(info)
    return rows


def dump_full_scripts():
    """把 4 个 D pod 的完整 start_command.sh 落盘到 build/output/，便于核对/diff。

    用用户真实场景的 env（平台动态分配 ephemeral 端口）。修复生效后，这 4 个脚本里的
    --data-parallel-rpc-port 应已被归一为角色固定常量（12777），即 wings 自愈后的结果。
    """
    out_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "build", "output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*78}\n落盘完整启动脚本（用户真实 env：平台按 pod 动态分配端口）\n{'='*78}")
    for idx, ip in enumerate(D_PODS):
        cmd = gen_pod_cmd(ip, DYNAMIC_RPC[idx])
        fn = f"start_command_pd-qwen3-4d1n-D_node{idx}.sh"
        path = os.path.join(out_dir, fn)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(cmd)
        info = extract(cmd)
        print(f"  build/output/{fn}  (rank {info['ranks']}, "
              f"rpc-port {info['rpc']}, address {info['addr']})")


def main():
    print("复现拓扑：D = 单节点 4 pod，DP8×TP1，local=2；head =", HEAD)

    rows_a = run_mode("A：consistent（正确基线）", lambda i: CONSISTENT_RPC)
    ports_a = {r["rpc"] for r in rows_a}
    addrs_a = {r["addr"] for r in rows_a}
    print(f"\n  → rpc-port 去重集合 = {ports_a}；address 去重集合 = {addrs_a}")
    print(f"  → {'✅ 4 pod rpc/address 全一致，可成组' if len(ports_a)==1 and len(addrs_a)==1 else '❌ 不一致'}")

    rows_b = run_mode("B：dynamic（平台按 pod 分配 ephemeral 端口，模拟故障）",
                      lambda i: DYNAMIC_RPC[i])
    ports_b = {r["rpc"] for r in rows_b}
    addrs_b = {r["addr"] for r in rows_b}
    print(f"\n  → rpc-port 去重集合 = {ports_b}；address 去重集合 = {addrs_b}")

    print(f"\n{'='*78}\n结论\n{'='*78}")
    if len(ports_b) > 1:
        print("❌ wings 原样透传：4 个 D pod 的 --data-parallel-rpc-port 各不相同，")
        print("   且 build 过程无任何报错/告警 → 非 head 的 rank 会连 head:<自己那份端口>，")
        print("   head 并未在该端口 listen → ZMQ 握手 5 分钟超时（与线上现象一致）。")
        print("   即：wings 不对 rpc-port 跨 pod 一致性做兜底，问题确实存在。")
    else:
        print("（dynamic 模式未产生不一致 → 兜底已生效）")

    dump_full_scripts()


if __name__ == "__main__":
    main()
